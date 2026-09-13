// Build-time rendering keeps the delivered book self-contained and offline.
const katex = require('./vendor/katex/katex.js');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => input += chunk);
process.stdin.on('end', () => {
  const rendered = JSON.parse(input).map(({tex, display}) =>
    katex.renderToString(tex, {
      output: 'mathml', displayMode: display,
      throwOnError: true, strict: 'ignore', trust: false,
    }));
  process.stdout.write(JSON.stringify(rendered));
});
