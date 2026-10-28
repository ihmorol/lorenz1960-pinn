const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

async function main() {
  const root = __dirname;
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({
      viewport: { width: 3179, height: 4494 },
      deviceScaleFactor: 150 / 96,
    });
    await page.goto(pathToFileURL(path.join(root, "poster.html")).href, { waitUntil: "load" });
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map((img) => img.decode()));
    });

    const qa = await page.evaluate(() => {
      const bands = [...document.querySelectorAll(".band")];
      const regions = [...document.querySelectorAll(
        ".band, .timeline, .inside, .step, .handoff, .evaluation, .problem-text, .baseline, .result-figure, .result-data, .small-figure, .limits"
      )];
      const overflow = regions.filter((node) => node.scrollHeight > node.clientHeight + 2)
        .map((node) => ({
          name: node.className,
          scroll: node.scrollHeight,
          client: node.clientHeight,
        }));
      const images = [...document.images].map((img) => {
        const vector = img.getAttribute("src").endsWith(".svg");
        return {
          src: img.getAttribute("src"),
          ok: img.naturalWidth > 0,
          vector,
          effectiveDpi: vector ? null : Math.round(img.naturalWidth * 96 / img.getBoundingClientRect().width),
        };
      });
      return {
        width: document.body.scrollWidth,
        height: document.body.scrollHeight,
        posterBottom: Math.round(document.querySelector(".poster").getBoundingClientRect().bottom),
        bandBottoms: bands.map((band) => Math.round(band.getBoundingClientRect().bottom)),
        overflow,
        fontsLoaded: document.fonts.check('400 40px "Source Sans 3"'),
        images,
      };
    });
    fs.writeFileSync(path.join(root, "qa.json"), JSON.stringify(qa, null, 2) + "\n");
    await page.screenshot({ path: path.join(root, "poster-a0-150dpi.png"), fullPage: true });
    if (qa.width !== 3179 || qa.height !== 4494 || qa.posterBottom !== 4494 ||
        qa.overflow.length || !qa.fontsLoaded ||
        qa.images.some((img) => !img.ok || (!img.vector && img.effectiveDpi < 150))) {
      throw new Error("Poster layout QA failed: " + JSON.stringify(qa));
    }
    await page.pdf({
      path: path.join(root, "poster-a0.pdf"),
      width: "841mm", height: "1189mm",
      margin: { top: "0mm", right: "0mm", bottom: "0mm", left: "0mm" },
      printBackground: true, preferCSSPageSize: true,
    });
    console.log(JSON.stringify(qa, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
