const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

async function main() {
  const root = __dirname;
  const browser = await chromium.launch({ headless: true });
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
    const panels = [...document.querySelectorAll(".panel")];
    const regions = [...document.querySelectorAll(".header, .metric, .thesis")];
    const regionOverflow = regions.filter((region) => region.scrollHeight > region.clientHeight + 2 || region.scrollWidth > region.clientWidth + 2).map((region) => ({ name: region.className, scroll: region.scrollHeight, client: region.clientHeight, width: region.scrollWidth, clientWidth: region.clientWidth }));
    const overflow = panels.filter((panel) => {
      const body = panel.querySelector(".panel-body");
      return panel.scrollHeight > panel.clientHeight + 2 || body.scrollHeight > body.clientHeight + 2;
    }).map((panel) => panel.querySelector("h2").textContent);
    const bottom = [...document.querySelectorAll(".row-three .panel")].map((panel) => Math.round(panel.getBoundingClientRect().bottom));
    return {
      width: document.body.scrollWidth,
      height: document.body.scrollHeight,
      posterBottom: Math.round(document.querySelector(".poster").getBoundingClientRect().bottom),
      bottom,
      overflow: [...regionOverflow, ...overflow],
      images: [...document.images].map((img) => ({ src: img.getAttribute("src"), ok: img.naturalWidth > 0, effectiveDpi: Math.round(img.naturalWidth * 96 / img.getBoundingClientRect().width) })),
    };
  });
  fs.writeFileSync(path.join(root, "qa.json"), JSON.stringify(qa, null, 2) + "\n");
  if (qa.width !== 3179 || qa.height !== 4494 || qa.posterBottom !== 4494 || qa.overflow.length || qa.images.some((img) => !img.ok || img.effectiveDpi < 150)) {
    await browser.close();
    throw new Error(`Poster layout QA failed: ${JSON.stringify(qa)}`);
  }

  await page.screenshot({ path: path.join(root, "poster-a0-150dpi.png"), fullPage: true });
  await page.pdf({ path: path.join(root, "poster-a0.pdf"), width: "841mm", height: "1189mm", margin: { top: "0mm", right: "0mm", bottom: "0mm", left: "0mm" }, printBackground: true, preferCSSPageSize: true });
  console.log(JSON.stringify(qa, null, 2));
  await browser.close();
}

main().catch((error) => { console.error(error); process.exit(1); });
