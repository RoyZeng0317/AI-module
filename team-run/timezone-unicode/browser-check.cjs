const {chromium} = require('C:/Users/roy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('assert');
const fs = require('fs');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const results = [];
  try {
    for (const timezoneId of ['Asia/Taipei', 'Asia/Shanghai', 'Asia/Kathmandu', 'America/New_York']) {
      const context = await browser.newContext({timezoneId, viewport: {width: 1280, height: 900}});
      const page = await context.newPage();
      const errors = [], requests = [];
      page.on('pageerror', e => errors.push(String(e)));
      await page.route('**/api/**', async route => {
        const req = route.request(), endpoint = new URL(req.url()).pathname.split('/api/')[1];
        requests.push(endpoint);
        let data = {};
        if (endpoint === 'conversations') data = req.method() === 'POST' ? {id: '1', title: '新對話', messages: []} : [];
        if (endpoint === 'commands') data = {commands: [], model_modes: ['auto','sinco','code','nvidia']};
        if (endpoint === 'gpu_info') data = {available: false};
        await route.fulfill({json: data});
      });
      await page.goto('http://127.0.0.1:8765/web/admin/frontend/src/components/index.html');
      await page.locator('#chatInput').waitFor({timeout: 60000});
      await page.getByText('後端已連線', {exact: true}).waitFor({state: 'attached'});
      await page.getByText('工具與個人化', {exact: true}).click();
      const before = requests.length;
      await page.locator('#unicodeToolBtn').click();
      await page.locator('#unicodeToolModal').waitFor({state: 'visible'});
      const info = await page.locator('#timezoneInfo').innerText();
      assert(info.includes(timezoneId) || (timezoneId === 'Asia/Kathmandu' && info.includes('Asia/Katmandu')), info);
      if (timezoneId === 'Asia/Taipei' || timezoneId === 'Asia/Shanghai') assert(/UTC\+0?8(?::?00)?/.test(info), info);
      if (timezoneId === 'Asia/Kathmandu') assert(/UTC\+0?5:45/.test(info), info);
      await page.locator('#unicodeInput').fill('中😀e\u0301<script>alert(1)</script>');
      await page.waitForFunction(() => document.querySelector('#unicodeResult').textContent.includes('U+1F600'));
      const unicode = await page.locator('#unicodeResult').innerText();
      assert(unicode.includes('U+4E2D') && unicode.includes('U+0301'), unicode);
      assert(await page.locator('#unicodeResult script').count() === 0);
      await page.locator('#unicodeInput').fill('😀'.repeat(201));
      await page.waitForTimeout(100);
      const long = await page.locator('#unicodeResult').innerText();
      assert((long.match(/U\+1F600/g) || []).length === 200, 'Limit must count Unicode code points');
      await page.locator('#unicodeInput').fill('台灣 中国 😀 e\u0301');
      await page.waitForTimeout(100);
      await page.screenshot({path: path.join(__dirname, timezoneId.replace('/', '-') + '.png')});
      if (timezoneId === 'Asia/Taipei') {
        await page.setViewportSize({width: 390, height: 844});
        await page.screenshot({path: path.join(__dirname, 'mobile.png')});
        assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)));
      }
      assert(requests.length === before, 'Tool must not upload text or fetch analysis APIs');
      await page.locator('#unicodeToolModal .overlay-close').click();
      assert(!(await page.locator('#unicodeToolModal').isVisible()));
      assert(errors.length === 0, JSON.stringify(errors));
      results.push({timezoneId, info, codePointLimit: 200, requestsDuringTool: requests.length - before, errors});
      await context.close();
    }
    fs.writeFileSync(path.join(__dirname, 'browser-results.json'), JSON.stringify(results, null, 2));
    console.log('PASS: four browser timezones, Unicode/emoji/combining marks, safe HTML, 200 code points, no uploads, mobile layout');
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exitCode = 1;});
