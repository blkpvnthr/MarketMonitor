import http from "http";
import fs from "fs";
import path from "path";
import url from "url";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const PORT = 5173;

const routes = {
  "/": path.join(__dirname, "public"),
  "/data_store": path.join(__dirname, "data_store"),
};

function serveFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }

    const ext = path.extname(filePath);
    const type = {
      ".html": "text/html",
      ".js": "application/javascript",
      ".css": "text/css",
      ".json": "application/json",
      ".csv": "text/csv",
    }[ext] || "application/octet-stream";

    res.writeHead(200, { "Content-Type": type });
    res.end(data);
  });
}

http.createServer((req, res) => {
  const parsed = url.parse(req.url);
  let pathname = parsed.pathname || "/";

  for (const [route, dir] of Object.entries(routes)) {
    if (pathname === route || pathname.startsWith(route + "/")) {
      let rel = pathname === route ? "/" : pathname.slice(route.length);
      let filePath = path.join(dir, rel);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, "index.html");
      }

      return serveFile(res, filePath);
    }
  }

  // default: public
  serveFile(res, path.join(__dirname, "public", pathname));
}).listen(PORT, () => {
  console.log(`✔ Server running at http://localhost:${PORT}`);
  console.log(`✔ UI:          http://localhost:${PORT}/confirmed-wall.html`);
  console.log(`✔ Data store:  http://localhost:${PORT}/data_store/`);
});