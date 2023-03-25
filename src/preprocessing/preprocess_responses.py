from typing import Dict, List
import requests
import pandas as pd
import re
import tqdm
from bs4 import BeautifulSoup
import base64
from loguru import logger
from src.config import paths

from src.reading.readers import read_raw_responses


def get_body(trimmed_message):
    body_start = re.search("\n\n", trimmed_message)
    body_end = re.search("\n--", trimmed_message)
    body_start_idx = body_start.end() if body_start is not None else None
    body_end_idx = body_end.start() if body_end is not None else None
    body = trimmed_message[body_start_idx:body_end_idx]
    return body


def get_next_part(row, column_w_response):
    trimmed_start = row[column_w_response][row["body_start_idx"] :]
    prebody_start = re.search(
        "Content-Transfer-Encoding: ", trimmed_start, re.IGNORECASE
    )
    if prebody_start is not None:
        trimmed = trimmed_start[prebody_start.end() :]
        body = get_body(trimmed)
    else:
        body = ""
    return body


def _get_hex_conversion_table(encoding_type="UTF-8"):
    """
    Downloads hex to char table, adds absent codes, returns df with code-char
    combinations.
    """
    url = "https://cs.wikipedia.org/wiki/K%C3%B3dov%C3%A1n%C3%AD_%C4%8De%C5%A1tiny"
    html = requests.get(url).content

    df_list = pd.read_html(html)
    codes = df_list[1]

    char = codes["Kódování"].Znak
    hex_code = codes[encoding_type].hex
    hex_code = "=" + hex_code.str.replace(" ", "=")
    conversion_df = pd.DataFrame({"hex_code": hex_code, "char": char})
    if encoding_type == "UTF-8":
        append = pd.DataFrame(
            {
                "hex_code": ["=20", "=E2=80=93", "=2E", "=2C", "=C2=A0", "=0A"],
                "char": [" ", "–", ".", ",", " ", " "],
            }
        )
        conversion_df = pd.concat([conversion_df, append], axis=0, ignore_index=True)
    elif encoding_type == "Windows-1250":
        append = pd.DataFrame(
            {
                "hex_code": ["=B9", "ľ"],
                "char": ["š", "ž"],
            }
        )
        conversion_df = pd.concat([conversion_df, append], axis=0, ignore_index=True)
    return conversion_df


def _fix_white_spaces(
    responses: pd.Series,
    replace_dict={"=\n": "", "\n": " ", "&nbsp;": " ", '"': "", "\r": ""},
):
    for key in replace_dict:
        responses = responses.str.replace(key, replace_dict[key])
    return responses


def _fix_encoding(responses: pd.Series, conversion_df: pd.DataFrame):
    for row in conversion_df.itertuples():
        responses = responses.str.replace(row.hex_code, row.char)
    return responses


def _get_entities_from_email_addr(email_adress, stopwords=["info"]):
    if isinstance(email_adress, str):
        name, domain = email_adress.split("@")
        names = re.split("\.|_|-", name) + [name]
        domain_clean = domain.split(".")[0]
        entities = list(pd.Series(names + [domain_clean]).unique())
        for word in stopwords:
            try:
                entities.remove(word)
            except:
                pass
        return entities
    else:
        return None


def extract_transfer_encoding(temp: pd.DataFrame, column_w_response: str):
    # extract transfer encoding and save the regex end index
    tr_encodings = []
    body_start_idx = []
    for i, row in temp.iterrows():
        body_start = re.search(
            "Content-Transfer-Encoding: ", row[column_w_response], re.IGNORECASE
        )
        if body_start is not None:
            trimmed_start = row[column_w_response][body_start.end() :]
            body_start_idx.append(body_start.end())
            tr_encodings.append(re.search(r"\S.*", trimmed_start)[0])
        else:
            body_start_idx.append("NULL")
            tr_encodings.append("NULL")
            temp["manual_check"].loc[i] = True
    return tr_encodings, body_start_idx


def extract_last_reply_based_on_transfer_encoding(
    temp: pd.DataFrame, column_w_response: str
):
    email_body = []
    for i, row in temp.iterrows():
        if row["tr_encodings"] == "binary" or row["tr_encodings"] == "7bit":
            body = get_next_part(row, column_w_response)
            email_body.append(body)
        elif row["tr_encodings"] == "8bit":
            trimmed_start = row[column_w_response][row["body_start_idx"] :]
            body = get_body(trimmed_start)
            email_body.append(body)
        elif row["tr_encodings"] == "base64":
            trimmed_start = row[column_w_response][row["body_start_idx"] :]
            body = get_body(trimmed_start)
            email_body.append(body)
        elif row["tr_encodings"] == "quoted-printable":
            trimmed_start = row[column_w_response][row["body_start_idx"] :]
            body = get_body(trimmed_start)
            email_body.append(body)
        else:
            body = ""
            email_body.append(body)
            temp["manual_check"].loc[i] = True
    return email_body


def decode_transfer_encoding(temp: pd.DataFrame):
    email_body_tr_decoded = []
    for i, row in temp.iterrows():
        if row["tr_encodings"] == "base64":
            if row["email_body"] != "":
                email_body_tr_decoded.append(
                    base64.b64decode(row["email_body"]).decode("utf-8")
                )
            else:
                email_body_tr_decoded.append("")
                temp["manual_check"].loc[i] = True
        else:
            email_body_tr_decoded.append(row["email_body"])
    return email_body_tr_decoded


def fix_html_tags(temp: pd.DataFrame):
    email_body_nohtml = []
    for i, row in tqdm.tqdm(temp.iterrows(), total=temp.shape[0]):
        email_body_nohtml.append(
            BeautifulSoup(row["email_body_win_fixed"], "lxml").text
        )
    return email_body_nohtml


def extract_body_of_reply(
    temp,
    start_possibilities: List[str] = ["Dobrý den,", "Dobré odpoledne,"],
    signoffs: List[str] = [
        "S pozdravem",
        "S pozdravom",
        "S přátelským pozdravem",
        "S pratelskym pozdravem",
        "S priatelskym pozdravom",
        "Zdravi[^a-zA-Z]",
        "Zdraví[^a-zA-Z]",
        "S pranim pekneho dne",
        "S přáním pěkného dne",
        "S pranim hezkeho dne",
        "S přáním hezkého dne",
        "S pranim krasneho dne",
        "S přáním krásného dne",
        "Preji pekny den",
        "Přeji pěkný den",
        "Preji hezky den",
        "Přeji hezký den",
        "Preji krasny den",
        "Přeji krásný den",
        "S uctou",
        "S úctou",
        "Dekuji",
        "Děkuji",
        "Dekuji za pochopeni",
        "Děkuji za pochopení",
        "Dekuju",
        "Děkuju",
        "Dekujeme",
        "Děkujeme",
        "Diky",
        "Díky",
        "Hezký den",
        "Hezky den",
        "Hezký zbytek dne",
        "Hezky zbytek dne",
        "Příjemný den",
        "Prijemny den",
        "Pěkný den",
        "Pekny den",
        "Hodne stesti",
        "Hodně štěstí",
        "Se srdečným pozdravem",
        "Se srdecnym pozdravem",
    ],
):
    body_clean = []
    for i, row in tqdm.tqdm(temp.iterrows(), total=temp.shape[0]):
        # search end of the first email
        end_idx = []
        for sign in signoffs:
            search = re.search(sign, row["email_body_strip"], re.IGNORECASE)
            idx = search.end() if search is not None else 0
            end_idx.append(idx)
        # discard signoffs that are too early
        end_idx = (
            pd.Series(end_idx)
            .loc[(pd.Series(end_idx) > 20) | (pd.Series(end_idx) == 0)]
            .tolist()
        )

        # search start of the possible second email
        start_idx = []
        for start in start_possibilities:
            if len(row["email_body_strip"]) > 5000:
                search = re.search("(?<!^)" + start, row["email_body_strip"])
                idx = search.end() if search is not None else 0
            else:
                search = re.search(".+(" + start + ".+)$", row["email_body_strip"])
                idx = search.span(1)[0] if search is not None else 0
            start_idx.append(idx)
        max_second_start_id = max(start_idx)

        # adjust the end of first email based on found start of second
        if max_second_start_id == 0:
            max_first_end_id = max(end_idx)
        else:
            max_first_end_id = max(
                pd.Series(end_idx).loc[pd.Series(end_idx) < max_second_start_id]
            )

        # if no end was found, try to search for signatures
        if max_first_end_id == 0:
            signatures = _get_entities_from_email_addr(row["from"])
            sign_idx = []
            if signatures is not None:
                for signature in signatures:
                    search = re.search(
                        signature, row["email_body_strip"], re.IGNORECASE
                    )
                    idx = search.end() if search is not None else 0
                    sign_idx.append(idx)
            # adjust the end of first email based on found start of second
            if max_second_start_id == 0:
                max_first_end_id = max(sign_idx)
            else:
                try:
                    max_first_end_id = max(
                        pd.Series(sign_idx).loc[
                            pd.Series(sign_idx) < max_second_start_id
                        ]
                    )
                except ValueError:
                    max_first_end_id = max_second_start_id
                if max_first_end_id == 0:
                    max_first_end_id = max_second_start_id

        # trimm the first email based on the last found signoff
        if max_first_end_id == 0:
            body_clean.append(row["email_body_strip"])
            temp["manual_check"].loc[i] = True
        else:
            body_clean.append(row["email_body_strip"][:max_first_end_id])

    return body_clean


def preprocess_responses(
    df_responses: pd.DataFrame,
    column_w_response: str,
    column_w_from_email: str,
    id_cols: List[str] = None,
    save_df_for_manual_check: bool = True,
    save_path: str = "data/preprocessed/responses_check.csv",
):
    """Preprocessing of emails.

    Extracts the body from last reply of an email.
    If there is something suspicious it flags the row to be checked manually.
    Around 97 % of emails preprocessed without any supervision needed, the rest
    3 % to be checked.
    Possible crashes:
        full email box
        foreign language (expects czech)
        broken/non-standard encoding
        empty responses
        last reply is not on top of the whole email
    """
    assert set([column_w_response, column_w_from_email]).issubset(
        df_responses.columns
    ), f"Columns needed for preprocessing - {column_w_response} and {column_w_from_email} - are missing"

    # delete failed emails
    if set(["failed"]).issubset(df_responses.columns):
        df_responses = df_responses.loc[~df_responses.failed]
        logger.info(f"Dropped failed emails, {len(df_responses)} rows left.")

    # copy the df with only columns needed for the preprocessing
    temp = df_responses[[*id_cols, column_w_response, column_w_from_email]]
    temp["manual_check"] = False

    # extract transfer encoding and save the regex end index
    logger.info("Extracting transfer encoding.")
    temp["tr_encodings"], temp["body_start_idx"] = extract_transfer_encoding(
        temp, column_w_response
    )

    # extract last reply based on transfer encoding
    logger.info("Extracting last replies.")
    temp["email_body"] = extract_last_reply_based_on_transfer_encoding(
        temp, column_w_response
    )

    # decode transfer encoding
    logger.info("Decoding the content based on transfer encoding.")
    temp["email_body_tr_decoded"] = decode_transfer_encoding(temp)

    # delete whitespace chars
    logger.info("Deleting whitespace characters.")
    temp["email_body_no_white"] = _fix_white_spaces(temp["email_body_tr_decoded"])

    # fix encoding of czech chars
    # utf-8
    logger.info("Fixing encoding of czech characters.")
    conversion_df_utf = _get_hex_conversion_table("UTF-8")
    temp["email_body_utf_fixed"] = _fix_encoding(
        temp["email_body_no_white"], conversion_df_utf
    )
    # windows-1250
    conversion_df_win = _get_hex_conversion_table("Windows-1250")
    temp["email_body_win_fixed"] = _fix_encoding(
        temp["email_body_utf_fixed"], conversion_df_win
    )

    # fix html tags
    logger.info("Deleting html tags.")
    temp["email_body_nohtml"] = fix_html_tags(temp)

    # delete extra spaces
    temp["email_body_strip"] = temp["email_body_nohtml"].str.split().str.join(" ")

    # search for sign-off
    logger.info("Extracting the body of last reply.")
    temp["body_clean"] = extract_body_of_reply(temp)
    temp["body_clean"]

    if save_df_for_manual_check:
        temp[
            [*id_cols, "manual_check", "body_clean", "email_body_strip", "full_text"]
        ].to_csv(save_path)

    return temp["body_clean"], temp["manual_check"]


def _save(df):
    df.to_csv(paths.DATA_PROCESSED_RESPONSES)
    logger.info(f"Preprocessed dataset saved at {paths.DATA_PROCESSED_RESPONSES}")
    pass


if __name__ == "__main__":

    pd.options.mode.chained_assignment = None

    # read the responses
    df_responses = read_raw_responses()
    column_w_response = "full_text"
    column_w_from_email = "from"
    id_cols = ["id", "id_2"]
    save_df_for_manual_check = True
    save_path = "data/preprocessed/emails_check_test.csv"

    # preprocess responses
    df_responses["text"], df_responses["manual_check"] = preprocess_responses(
        df_responses, column_w_response, column_w_from_email, id_cols
    )

    # save
    _save(df_responses)

    pass
