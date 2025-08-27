const btnValid = document.getElementById("btnValid");
const messageEl = document.getElementById("message");
const statusEl = document.getElementById("status");
const translateDiv = document.getElementById("translate")

btnValid.addEventListener("click", async () => {
    statusEl.textContent = "Analyse en cours...";
    btnValid.disabled = true;

    try {
        const translationResult = await fetch("/translate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({message:messageEl.value})
        })
        const translateJson = await translationResult.json();
        if(translateJson.translation) {

            translateDiv.innerText = translateJson.translation;
            statusEl.textContent = "Traduction :";
            btnValid.disabled = false;

        } else {
            statusEl.textContent = "Erreur dans le processus de traduction.";
            btnValid.disabled = false;
        }


    }catch (error) {
        alert("Erreur lors de l'envoi du message");
    }

})