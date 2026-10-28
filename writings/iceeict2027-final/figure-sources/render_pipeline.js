const path = require('path');
const { spawnSync } = require('child_process');
const { pathToFileURL } = require('url');
const { chromium } = require(path.join(process.argv[5], 'playwright'));

async function main() {
  const source = path.resolve(process.argv[2]);
  const pdfOutput = path.resolve(process.argv[3]);
  const svgOutput = path.resolve(process.argv[4]);
  const browser = await chromium.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1088, height: 2242 } });
  await page.goto(pathToFileURL(source).href, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForFunction(() => window.MathJax?.startup?.promise, null, { timeout: 30000 });
  await page.evaluate(() => Promise.race([
    window.MathJax.startup.promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('MathJax timeout')), 30000)),
  ]));
  await page.evaluate(() => document.fonts.ready);

  const validation = await page.evaluate(() => {
    const elements = [...document.querySelectorAll('.n, .tag')];
    const wireLabelCount = document.querySelectorAll('.wire text').length;
    const clipped = [...document.querySelectorAll('.n')]
      .filter((element) => element.scrollWidth > element.clientWidth + 1 || element.scrollHeight > element.clientHeight + 1)
      .map((element) => ({
        text: element.textContent.trim().replace(/\s+/g, ' '),
        client: [element.clientWidth, element.clientHeight],
        scroll: [element.scrollWidth, element.scrollHeight],
      }));
    const fontSizes = elements.map((element) => parseFloat(getComputedStyle(element).fontSize) * 72 / 96);
    const minimumSourceLabelPt = Math.min(27.75, ...fontSizes);
    return {
      widthPt: 816,
      heightPt: parseFloat(document.querySelector('.pipe').style.height),
      labels: elements.length + wireLabelCount,
      minimumSourceLabelPt,
      minimumPlacedLabelPt: minimumSourceLabelPt * 251.06 / 816,
      timesNewRomanAvailable: document.fonts.check('27.75pt "Times New Roman"'),
      clipped,
    };
  });
  if (!validation.timesNewRomanAvailable || validation.clipped.length) {
    throw new Error(JSON.stringify(validation));
  }

  await page.pdf({ path: pdfOutput, printBackground: true, preferCSSPageSize: true });
  await browser.close();
  const converter = spawnSync('python', [
    '-c',
    'import fitz,sys; d=fitz.open(sys.argv[1]); open(sys.argv[2],"w",encoding="utf-8").write(d[0].get_svg_image(text_as_path=False))',
    pdfOutput,
    svgOutput,
  ], { encoding: 'utf8' });
  if (converter.status !== 0) throw new Error(converter.stderr);
  process.stdout.write(JSON.stringify(validation, null, 2) + '\n');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
