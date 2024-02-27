[![NPM version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Dependency Status][gemnasium-image]][gemnasium-url]

# parse-accept-language

Parse 'Accept-Language' header and cache the result in request.

## Install

```sh
$ npm install --save parse-accept-language
```

## Usage

```js
var parseAcceptLanguage = require('parse-accept-language');

var pal = parseAcceptLanguage(req);
```

## License

MIT © [Damian Krzeminski](https://pirxpilot.me)

[npm-image]: https://img.shields.io/npm/v/parse-accept-language.svg
[npm-url]: https://npmjs.org/package/parse-accept-language

[travis-url]: https://travis-ci.org/pirxpilot/parse-accept-language
[travis-image]: https://img.shields.io/travis/pirxpilot/parse-accept-language.svg

[gemnasium-image]: https://img.shields.io/gemnasium/pirxpilot/parse-accept-language.svg
[gemnasium-url]: https://gemnasium.com/pirxpilot/parse-accept-language
