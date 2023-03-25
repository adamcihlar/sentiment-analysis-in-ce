class Mall:
    models = [
        ("seznamsmall-e-czech_20230123-173904", "5", "facebook"),
        ("seznamsmall-e-czech_20230123-181226", "5", "csfd"),
        ("seznamsmall-e-czech_20230128-144144", "4", "facebook"),
        ("seznamsmall-e-czech_20230128-144144", "5", "csfd"),
        ("seznamsmall-e-czech_20230218-183829", "5", "facebook_csfd"),
    ]
    temperatures = [1, 2, 5, 10, 20]
    loss_combination_params_list = [
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
    ]


class CSFD:
    models = [
        ("seznamsmall-e-czech_20230123-173904", "5", "facebook"),
        ("seznamsmall-e-czech_20230123-213412", "5", "mall"),
        ("seznamsmall-e-czech_20230128-171700", "5", "facebook"),
        ("seznamsmall-e-czech_20230128-171700", "5", "mall"),
        ("seznamsmall-e-czech_20230218-142950", "5", "mall_facebook"),
    ]
    temperatures = [1, 2, 5, 10, 20]
    loss_combination_params_list = [
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
    ]


class Facebook:
    models = [
        ("seznamsmall-e-czech_20230123-181226", "5", "csfd"),
        ("seznamsmall-e-czech_20230123-213412", "5", "mall"),
        ("seznamsmall-e-czech_20230128-055746", "5", "csfd"),
        ("seznamsmall-e-czech_20230128-055746", "5", "mall"),
        ("seznamsmall-e-czech_20230218-213739", "5", "mall_csfd"),
    ]
    temperatures = [1, 2, 5, 10, 20]
    loss_combination_params_list = [
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
    ]
