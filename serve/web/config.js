// Where the model runs. The FastAPI server (serve/api.py) serves this file as it is:
// "server" mode, the page calls the server's /v1 API.
// The static site (deploy/build_static.py) replaces it with "browser" mode and the model's
// URLs: the model is then downloaded and run inside the visitor's browser (engine.js).
window.TL_CONFIG = { mode: "server" };
