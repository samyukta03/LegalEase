[![NPM version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Dependency Status][deps-image]][deps-url]
[![Dev Dependency Status][deps-dev-image]][deps-dev-url]

# detect-language

Finds the best matching language from Accept-Language header.

## Install

```sh
$ npm install --save detect-language
```

## Usage

```js
var app = require('express');

var locale = {
  supportedLanguages: ['de', 'fr', 'pl', 'en-GB', 'en-US'],
  defaultLanguage: 'en'
};

app.use(require('./src/i18n/detect-language')(locale));

// req.lang is set to detected language

```

## License

MIT © [Damian Krzeminski](https://pirxpilot.me)

[npm-image]: https://img.shields.io/npm/v/detect-language.svg
[npm-url]: https://npmjs.org/package/detect-language

[travis-url]: https://travis-ci.org/pirxpilot/detect-language
[travis-image]: https://img.shields.io/travis/pirxpilot/detect-language.svg

[deps-image]: https://img.shields.io/david/pirxpilot/detect-language.svg
[deps-url]: https://david-dm.org/pirxpilot/detect-language

[deps-dev-image]: https://img.shields.io/david/dev/pirxpilot/detect-language.svg
[deps-dev-url]: https://david-dm.org/pirxpilot/detect-language?type=dev
