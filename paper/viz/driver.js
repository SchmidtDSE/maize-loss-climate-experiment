let globalTooltips = null;
let globalTabs = null;
let globalSlider = null;


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
    globalTabs = new Tabby("[data-tabs]");

    document.addEventListener("tabby", function (event) {
        const url = new URL(event.target.href);
        window.location.hash = url.hash;
        window.scrollTo(0, 0);
    }, false);

    const tabAdvanceLinks = document.querySelectorAll(".advance-button");
    tabAdvanceLinks.forEach((x) => x.addEventListener("click", (event) => {
        const tabName = x.getAttribute("tab");
        globalTabs.toggle(tabName);
        window.location.hash = "#" + tabName;
        window.scrollTo(0, 0);
        event.preventDefault();
    }));

    return globalTabs;
}


function initSlider() {
    globalSlider = tns({
        container: ".intro-slider",
        items: 1,
        slideBy: "page",
        autoplay: false,
        nav: false
    });

    const slideAdvanceLinks = document.querySelectorAll(".slide-advance-link");
    slideAdvanceLinks.forEach((x) => x.addEventListener("click", (event) => {
        if (globalSlider === null) {
            return;
        }
        
        globalSlider.goTo("next");
        document.getElementById("intro-slider").scrollIntoView({"behavior": "smooth", "block": "start"});
        event.preventDefault();
    }));

    document.getElementById("model-skip-link").addEventListener("click", (event) => {
        if (globalSlider === null) {
            return;
        }
        
        globalSlider.goTo(3);
        document.getElementById("intro-slider").scrollIntoView({"behavior": "smooth", "block": "start"});
        document.getElementById("model-overview").focus();
        event.preventDefault();
    });

    document.getElementById("finish-slides-link").addEventListener("click", (event) => {
        if (globalSlider === null) {
            return;
        }
        
        globalSlider.goTo("last");
        document.getElementById("intro-slider").scrollIntoView({"behavior": "smooth", "block": "start"});
        document.getElementById("finish-slide").focus();
        event.preventDefault();
    });

    return globalSlider;
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

    document.getElementById("toc-link").addEventListener("click", (event) => {
        globalTabs.toggle("introduction");
        document.getElementById("toc").setAttribute("open", true);
        document.getElementById("toc").focus();
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

    const vizRadios = document.querySelectorAll(".visualizations-setting-radio");
    vizRadios.forEach((x) => x.addEventListener("change", function () {
        if (this.checked) {
            const defaultVizs = document.querySelectorAll(".default-viz");
            const keyboardControls = document.querySelectorAll(".keyboard-controls");
            const vizAlternatives = document.querySelectorAll(".viz-alternative");
            const keyboardSettings = document.getElementById("keyboard-setting");

            const hide = (x) => x.style.display = "none";
            const show = (x) => x.style.display = "block";

            if (this.value === "hide") {
                defaultVizs.forEach(hide);
                keyboardControls.forEach((x) => x.classList.add("override"));
                vizAlternatives.forEach(show);
                hide(keyboardSettings);
            } else {
                defaultVizs.forEach(show);
                keyboardControls.forEach((x) => x.classList.remove("override"));
                vizAlternatives.forEach(hide);
                show(keyboardSettings);
            }
        }
    }));

    const keyboardRadios = document.querySelectorAll(".keyboard-setting-radio");
    keyboardRadios.forEach((x) => x.addEventListener("change", function () {
        if (this.checked) {
            const keyboardControls = document.querySelectorAll(".keyboard-controls");

            const hide = (x) => x.classList.remove("visible");
            const show = (x) => x.classList.add("visible");

            if (this.value === "hide") {
                keyboardControls.forEach(hide);
            } else {
                keyboardControls.forEach(show);
            }
        }
    }));

    const sliderRadios = document.querySelectorAll(".slider-setting-radio");
    sliderRadios.forEach((x) => x.addEventListener("change", function () {
        if (this.checked) {
            if (this.value === "show") {
                initSlider();
            } else {
                globalSlider.destroy();
                globalSlider = null;
            }
        }
    }));

    const tooltipRadios = document.querySelectorAll(".tooltips-setting-radio");
    tooltipRadios.forEach((x) => x.addEventListener("change", function () {
        if (this.checked) {
            const tooltipElems = document.querySelectorAll(".tooltip");
            if (this.value === "show") {
                globalTooltips.forEach((x) => x.enable());
                tooltipElems.forEach((x) => x.classList.remove("disabled"));
                tooltipElems.forEach((x) => x.setAttribute("tabindex", 0));
            } else {
                globalTooltips.forEach((x) => x.disable());
                tooltipElems.forEach((x) => x.classList.add("disabled"));
                tooltipElems.forEach((x) => x.removeAttribute("tabindex"));
            }
        }
    }));

    const canvases = document.querySelectorAll(".focus-canvas");
    canvases.forEach((canvas) => {
        canvas.addEventListener('keyup', (event) => {
            const isEscape = event.key === "Escape" || event.key === "Esc";
            if (isEscape || isTab) {
                canvas.blur();
            }
        });
    });
}


function main() {
    globalTooltips = tippy("[data-tippy-content]");
    globalTabs = initTabs();
    globalSlider = initSlider();
    initInteractivesLinks();
    initBespokeControls();
    initAccessibility();
}


main();
