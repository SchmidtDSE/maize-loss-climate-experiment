/**
 * Start the web application.
 */
function openWebApp(event) {
    document.querySelectorAll(".app-intro").forEach((x) => x.style.display = "none");
    document.querySelectorAll(".sketch").forEach((x) => x.style.display = "block");

    const execute = () => {
        const scriptTag = document.createElement("script");
        scriptTag.type = "py";
        scriptTag.src = "web.pyscript?v=" + Date.now();
        document.getElementById("root").appendChild(scriptTag);
    };

    let progress = 0;
    const progressBars = document.querySelectorAll(".sketch-load-progress");
    progressBars.forEach((x) => x.value = 0);
    const incrementBar = () => {
        let updateWaiting = false;
        
        progressBars.forEach((progressBar) => {
            progressBar.value += 1;
            updateWaiting = updateWaiting || progressBar.value < 19;
        });

        if (updateWaiting) {
            setTimeout(incrementBar, 500);
        }
    };

    event.preventDefault();

    incrementBar();
    execute();
}


function initTabs() {
    const tabs = new Tabby("[data-tabs]");

    document.addEventListener("tabby", function (event) {
        const url = new URL(event.target.href);
        window.location.hash = url.hash;
        window.scrollTo(0, 0);
    }, false);

    const tabAdvanceLinks = document.querySelectorAll(".advance-button");
    tabAdvanceLinks.forEach((x) => x.addEventListener("click", (event) => {
        const tabName = x.getAttribute("tab");
        tabs.toggle(tabName);
        window.location.hash = "#" + tabName;
        window.scrollTo(0, 0);
        event.preventDefault();
    }));
}


function initSlider() {
    const slider = tns({
        container: ".intro-slider",
        items: 1,
        slideBy: "page",
        autoplay: false,
        nav: false
    });

    const slideAdvanceLinks = document.querySelectorAll(".slide-advance-link");
    slideAdvanceLinks.forEach((x) => x.addEventListener("click", (event) => {
        slider.goTo("next");
        document.getElementById("#intro-slider").scrollIntoView({"behavior": "smooth", "block": "start"});
        event.preventDefault();
    }));

    document.getElementById("model-skip-link").addEventListener("click", (event) => {
        slider.goTo(3);
        event.preventDefault();
        document.getElementById("#intro-slider").scrollIntoView({"behavior": "smooth", "block": "start"});
        document.getElementById("model-overview").focus();
    });

    document.getElementById("finish-slides-link").addEventListener("click", (event) => {
        slider.goTo("last");
        event.preventDefault();
        document.getElementById("#intro-slider").scrollIntoView({"behavior": "smooth", "block": "start"});
        document.getElementById("finish-slide").focus();
    });
}


function initInteractivesLinks() {
    const loadLinks = document.querySelectorAll(".load-app-link");
    loadLinks.forEach((x) => x.addEventListener("click", openWebApp));
}


function initBespokeControls() {
    document.getElementById("expand-rate-var-link").addEventListener("click", (event) => {
        document.getElementById("rate-var-details").setAttribute("open", true);
        document.getElementById("rate-var-details").focus();
        event.preventDefault();
    });
}


function initAccessibility() {
    const symbolRadios = document.querySelectorAll(".symbols-setting-radio");
    symbolRadios.forEach((x) => x.addEventListener("change", function () {
        if (this.checked) {
            const emojis = document.querySelectorAll(".emoji");
            const chverons = document.querySelectorAll(".chevron");

            const hide = (x) => x.style.display = "none";
            const show = (x) => x.style.display = "inline-block";

            if (this.value === "hide") {
                emojis.forEach(hide);
                chverons.forEach(hide);
            } else {
                emojis.forEach(show);
                chverons.forEach(show);
            }
        }
    }));
}


function main() {
    tippy("[data-tippy-content]");

    initTabs();
    initSlider();
    initInteractivesLinks();
    initBespokeControls();
    initAccessibility();
}


main();
