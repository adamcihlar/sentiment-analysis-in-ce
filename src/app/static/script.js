anime({
	targets: '.card',
	translateY: [50, 0],
	opacity: [0, 1],
	duration: 2000,
	delay: anime.stagger(200, {start: 200})
})

function copyInvite() {
	var copyText = document.getElementById("invite-link");
	alert(copyText.value);
}
