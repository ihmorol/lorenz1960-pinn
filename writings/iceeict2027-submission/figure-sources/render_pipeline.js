const { pathToFileURL } = require('url');
const path = require('path');
const { chromium } = require(path.join(process.argv[4], 'playwright'));

async function main() {
  const source = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  const browser = await chromium.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1632, height: 2340 } });
  await page.goto(pathToFileURL(source).href, { waitUntil: 'networkidle' });
  await page.evaluate(async () => {
    await document.fonts.ready;
    if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise;
  });
  await page.pdf({ path: output, printBackground: true, preferCSSPageSize: true });
  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
