/* Real Chromium DOM/execution checks, deliberately without CSP.
Run fixtures first with xss_fixtures.py, then node test_xss_dom.cjs <fixtures.json>.
Set NODE_PATH to an installed playwright package directory when needed. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '../../../../..');
const fixture = JSON.parse(fs.readFileSync(process.argv[2] || path.join(root, 'output/xss-fixtures.json'), 'utf8').replace(/^\uFEFF/, ''));
(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.route('https://xss.test/**', route => route.fulfill({contentType:'text/html',body:'<!doctype html><body></body>'}));
    await page.goto('https://xss.test/');
    // Reproduce the original label sink, proving this browser executes onerror without CSP.
    await page.evaluate(() => { window.__xss=0; document.body.innerHTML='<span class="dot"></span><img src=x onerror="window.__xss=1"> × 1'; });
    await page.waitForFunction(() => window.__xss===1);
    const js = fs.readFileSync(path.join(root,'web/admin/frontend/script.js'),'utf8');
    const markdown = js.slice(js.indexOf('function escapeHtml'),js.indexOf('// fetch()'));
    const draw = js.slice(js.indexOf('  function drawDetections'), js.lastIndexOf('\n}\n'));
    await page.evaluate(({fixture, markdown, draw}) => {
      document.body.replaceChildren(); window.__xss=0;
      (0,eval)(markdown);
      for (const test of fixture.cases) {
        const node=document.createElement('section'); node.dataset.test=test.name; node.innerHTML=test.html; document.body.append(node);
      }
      for (const payload of fixture.payloads) {
        const node=document.createElement('section'); node.dataset.test='legacy-js-markdown'; node.innerHTML=markdownToHtml(payload); document.body.append(node);
      }
      window.resultsEl=document.createElement('div'); document.body.append(resultsEl);
      window.overlay={width:100,height:100}; window.ctx=new Proxy({}, {get:(_, key)=>key==='measureText'?()=>({width:1}):()=>{}});
      (0,eval)(draw);
      for (const payload of fixture.payloads) { drawDetections({width:100,height:100,detections:[{label:payload,confidence:1,box:[0,0,1,1]}]});
        const copy=resultsEl.cloneNode(true); copy.dataset.test='legacy-js-label'; document.body.append(copy); }
    }, {fixture,markdown,draw});
    await page.waitForTimeout(150);
    const violations=await page.evaluate(() => [...document.querySelectorAll('*')].flatMap(el=>{
      const issues=[];
      if (el.tagName==='A' && [...el.attributes].some(attr=>!['href','target','rel'].includes(attr.name))) issues.push(el.outerHTML);
      if (/^(SCRIPT|IFRAME|OBJECT|EMBED|IMG)$/.test(el.tagName.toUpperCase())) issues.push(el.outerHTML);
      for (const attr of el.attributes) if (/^on/i.test(attr.name)||['srcdoc'].includes(attr.name)||(['href','src'].includes(attr.name)&&!/^https?:/i.test(el[attr.name]))) issues.push(el.outerHTML);
      return issues;
    }));
    assert.deepEqual(violations,[]);
    assert.equal(await page.evaluate(()=>window.__xss),0);
    assert.equal(await page.locator('[data-test="valid-link"] a').getAttribute('href'),'https://example.com/a_b?q=1&x=2');
    assert.equal(await page.locator('[data-test="valid-link"] code').textContent(),'**literal**');
    const oddHref=await page.locator('[data-test$="action.py:markdown:8"] a').getAttribute('href');
    assert.equal(oddHref,'https://example.com/`onmouseover=window.__xss=1`');
    // URL policy is checked independently from CSP and same-origin alone.
    const imageJS=fs.readFileSync(path.join(root,'web/image_processing/static/app.js'),'utf8');
    const localURL=imageJS.slice(imageJS.indexOf('function localURL'),imageJS.indexOf('async function jsonResponse'));
    const urlChecks=await page.evaluate(source=>{
      (0,eval)(source);
      const good='/api/jobs/'+'a'.repeat(32)+'/files/input.png?download=1';
      const bad=['javascript:window.__xss=1','data:text/html,x','blob:https://xss.test/id','https://evil.test'+good,'/api/chat','/api/jobs/'+'a'.repeat(32)+'/files/a.svg','/api/jobs/'+'a'.repeat(32)+'/files/%22.png',good+'#x',good+'&x=1'];
      return {good:localURL(good),bad:bad.map(value=>{try {localURL(value);return value;} catch {return null;}})};
    },localURL);
    assert.equal(urlChecks.good,'https://xss.test/api/jobs/'+'a'.repeat(32)+'/files/input.png?download=1');
    assert.ok(urlChecks.bad.every(value=>value===null));
    const imageResults=await page.evaluate(({source,payloads})=>{
      const container=document.createElement('section'); document.body.append(container);
      for (const id of ['before-image','after-image','before-label','after-label','before-caption','after-caption','summary','histograms','download-list','result-details']) {
        const node=document.createElement('div');node.id=id;container.append(node);
      }
      const renderer=new Function(source+'; return render;')();
      for (const payload of payloads) {
        renderer({input:{width:1,height:1,name:payload},files:[{name:payload,label:payload,download_url:'/api/jobs/'+'a'.repeat(32)+'/files/raw.json?download=1'}]},1,'convolution',{channel:'rgb',display:'minmax'});
        if (container.querySelector('script,img,svg,iframe') || container.querySelector('[onerror],[onload],[onclick]')) return false;
        if (!document.getElementById('before-caption').textContent.includes(payload)) return false;
        if (!document.getElementById('download-list').textContent.includes(payload)) return false;
      }
      return window.__xss===0;
    },{source:imageJS.slice(0,imageJS.indexOf('async function processFile')),payloads:fixture.payloads});
    assert.equal(imageResults,true);
    console.log(JSON.stringify({browser:await browser.version(),fixtureCases:fixture.cases.length,jsPayloadCases:fixture.payloads.length*2,preFixOnerrorExecuted:true,postFixExecuted:false,urlRejections:urlChecks.bad.length,imageFilenameCases:fixture.payloads.length,CSP:false}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
