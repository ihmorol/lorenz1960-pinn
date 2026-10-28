const { chromium } = require('playwright');
const path = require('path');
const page = 'file://' + path.resolve(__dirname, '../final/poster.html');
(async () => {
  const b = await chromium.launch();
  for (const [q, out] of [['', 'poster.pdf'], ['?font=new', 'poster-newfont.pdf']]) {
    const p = await b.newPage();
    await p.goto(page + q, { waitUntil: 'networkidle' });
    await p.evaluate(async () => { await MathJax.startup.promise; await document.fonts.ready; });
    const tight = await p.evaluate(() => [...document.querySelectorAll('.card')]
      .filter(c => c.lastElementChild.getBoundingClientRect().bottom > c.getBoundingClientRect().bottom - 4)
      .map(c => c.querySelector('h2').textContent));
    if (tight.length) console.log(out, 'overflowing cards:', tight.join(', '));
    await p.pdf({ path: path.resolve(__dirname, '../final', out), preferCSSPageSize: true, printBackground: true });
    await p.close();
  }
  const p = await b.newPage({ deviceScaleFactor: 3 });
  await p.goto('file://' + path.resolve(__dirname, '../final/pipeline.html'), { waitUntil: 'networkidle' });
  await p.evaluate(async () => { await MathJax.startup.promise; await document.fonts.ready; });
  await p.pdf({ path: path.resolve(__dirname, '../final/pipeline.pdf'), preferCSSPageSize: true, printBackground: true });
  await p.locator('.pipe').screenshot({ path: path.resolve(__dirname, '../final/pipeline.png') });
  await b.close();
})();
