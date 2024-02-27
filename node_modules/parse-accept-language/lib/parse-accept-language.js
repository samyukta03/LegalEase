module.exports = parse;
module.exports.toArray = str2array;

function parseTag(tag) {
  tag = tag.split(';');
  const el = {
    value: tag[0].trim(),
    q: tag[1]
  };
  if (!el.value) {
    return;
  }

  const lang = el.value.split('-');

  el.language = lang[0];
  el.region = (lang[1] || '').toUpperCase();
  if (!el.q) {
    el.q = 1;
  } else {
    el.q = parseFloat(el.q.slice(2));
    if (isNaN(el.q)) {
      el.q = 1;
    }
  }
  return el;
}

function str2array(acceptLanguage) {
  return (acceptLanguage || '').split(',')
    .map(parseTag)
    .filter(Boolean)  // filter empty
    .sort((a, b) => b.q - a.q);
}

// parse Accept-Language header with memoization
function parse(req) {
  if(req._parsedAcceptLanguage) {
    return req._parsedAcceptLanguage;
  }
  const acceptLanguage = req.get('accept-language');
  req._parsedAcceptLanguage = str2array(acceptLanguage);
  req._parsedAcceptLanguage._raw = acceptLanguage;
  return req._parsedAcceptLanguage;
}
