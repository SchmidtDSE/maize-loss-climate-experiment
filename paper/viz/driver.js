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


function main() {
    tippy("[data-tippy-content]");

    const tabs = new Tabby("[data-tabs]");
    const slider = tns({
        container: ".intro-slider",
        items: 1,
        slideBy: "page",
        autoplay: false,
        nav: false
    });

    document.addEventListener("tabby", function (event) {
        const url = new URL(event.target.href);
        window.location.hash = url.hash;
        window.scrollTo(0, 0);
    }, false);

    const loadLinks = document.querySelectorAll(".load-app-link");
    loadLinks.forEach((x) => x.addEventListener("click", openWebApp));

    const advanceLinks = document.querySelectorAll(".advance-button");
    advanceLinks.forEach((x) => x.addEventListener("click", (event) => {
        const tabName = x.getAttribute("tab");
        tabs.toggle(tabName);
        window.location.hash = "#" + tabName;
        window.scrollTo(0, 0);
        event.preventDefault();
    }));

    document.getElementById("model-skip-link").addEventListener("click", (event) => {
        slider.goTo(3);
        event.preventDefault();
        document.getElementById("model-overview").focus();
    });

    document.getElementById("finish-slides-link").addEventListener("click", (event) => {
        slider.goTo("last");
        event.preventDefault();
        document.getElementById("finish-slide").focus();
    });
    
    document.getElementById("expand-rate-var-link").addEventListener("click", (event) => {
        document.getElementById("rate-var-details").setAttribute("open", true);
        document.getElementById("rate-var-details").focus();
        event.preventDefault();
    });
}


main();
