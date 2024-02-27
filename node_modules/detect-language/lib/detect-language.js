// middleware that selects preferred language from the list of defaults

const debug = require('debug')('detect-language');

// see: http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.4
function findMatch(supportedLanguages, parsedAcceptLanguage) {
  let candidate;

  for (let a = 0; a < parsedAcceptLanguage.length; a += 1) {
    candidate = null;
    const accepted = parsedAcceptLanguage[a];
    for (let s = 0; s < supportedLanguages.length; s += 1) {
      const supported = supportedLanguages[s];
      if (supported.value === accepted.value) {
        return supportedLanguages[s];
      }
      if (!candidate && supported.language === accepted.language) {
        candidate = supported;
      }
    }
    if (candidate) {
      return candidate;
    }
  }
}

function lang2obj(lang) {
  const codeAndRegion = lang.split('-');
  return {
    value: lang,
    language: codeAndRegion[0],
    region: codeAndRegion[1]
  };
}

const parseAcceptLanguage = require('parse-accept-language');

module.exports = function (opts) {
  const supportedLanguages = [ opts.defaultLanguage ]
    .concat(opts.supportedLanguages)
    .map(lang2obj);

  // mark language the we select by default with low quality
  const defaultParsedLanguage = Object.assign({ q: -1 }, supportedLanguages[0]);

  return function detectLanguage(req, res, next) {
    if (req.lang) {
      debug('skip - language already detected:', req.lang);
      return next();
    }
    debug('started:', req.lang);
    const pal = parseAcceptLanguage(req);
    req.parsedLang = findMatch(supportedLanguages, pal) || defaultParsedLanguage;
    req.lang = req.parsedLang.value;
    debug('detected:', req.lang);
    next();
  };
};
