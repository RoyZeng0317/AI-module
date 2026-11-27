const {chromium} = require('C:/Users/roy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('assert');
const fs = require('fs');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const results = [];
  try {
    for (const timezoneId of ['Asia/Shanghai', 'Asia/Kuala_Lumpur', 'Asia/Taipei']) {
      const context = await browser.newContext({timezoneId});
      const page = await context.newPage();
      const errors = [], sent = [];
      page.on('pageerror', e => errors.push(String(e)));
      await page.route('**/api/**', async route => {
        const req = route.request(), endpoint = new URL(req.url()).pathname.split('/api/')[1];
        let data = {};
        if (endpoint === 'conversations') data = req.method() === 'POST' ? {id: '1', title: '新對話', messages: []} : [];
        if (endpoint === 'commands') data = {commands: [], model_modes: ['auto','sinco','code','nvidia']};
        if (endpoint === 'chat') {sent.push(req.postDataJSON()); data = {reply: '收到測試訊息', trace: '一般回覆'};}
        await route.fulfill({json: data});
      });
      await page.goto('http://127.0.0.1:8765/web/admin/frontend/src/components/index.html');
      await page.locator('#chatInput').waitFor({timeout: 60000});
      await page.locator('#chatInput').fill('那咋啦，中国，测试');
      await page.locator('#chatInput').press('Enter');
      await page.getByText('收到測試訊息', {exact: true}).waitFor();
      assert(sent.length === 1 && sent[0].client_timezone === timezoneId, JSON.stringify(sent));
      assert(!('country' in sent[0]) && !('ip' in sent[0]), 'Client must not supply trusted geo facts');
      const text = await page.locator('body').innerText();
      assert(!/初步判定|二次判定|評判|內部判定|時區與 Unicode/.test(text), 'No visible region assessment');
      assert(await page.locator('#unicodeToolModal').count() === 0);
      assert(errors.length === 0, JSON.stringify(errors));
      results.push({timezoneId, sentTimezone: sent[0].client_timezone, visibleAssessment: false, errors});
      await context.close();
    }
    fs.writeFileSync(path.join(__dirname, 'browser-results.json'), JSON.stringify(results, null, 2));
    console.log('PASS: silent timezone payload for Shanghai/Kuala Lumpur/Taipei, no region UI, chat unchanged');
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exitCode = 1;});
