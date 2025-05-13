/**
 * @param {...(null|boolean|number|string|Array|Object)} args
 * @return {number}
 */
var argumentsLength = function (...args) {
  return arguments.length;
};

/**
 * @param {Function} fn
 * @return {Function}
 */
var once = function (fn) {
  let called = false;
  return function (...args) {
    if (called) return undefined;
    called = true;
    return fn(...args);
  };
};

let fn = (a, b, c) => a + b + c;
let onceFn = once(fn);

/**
 * @param {Promise} promise1
 * @param {Promise} promise2
 * @return {Promise}
 */
var addTwoPromises = async function (promise1, promise2) {
  // return (await promise1) + (await promise2);
  return await Promise.all([promise1, promise2]).then(([v1, v2]) => v1 + v2);
};

// addTwoPromises(Promise.resolve(2), Promise.resolve(2)).then(console.log); // 4
/**
 * @param {number} millis
 * @return {Promise}
 */
async function sleep(millis) {
  return new Promise((resolve) => setTimeout(resolve, millis));
}

// let t = Date.now();
// sleep(100).then(() => console.log(Date.now() - t)); // 100
/**
 * basically what we are doing is to create a function that will start executing something,
 * if its called before the time runs out it will cancel what its doing. so we return a clear function that can be called
 * @param {Function} fn
 * @param {Array} args
 * @param {number} t
 * @return {Function}
 */
var cancellable = function (fn, args, t) {
  const res = setTimeout(() => fn(...args), t);
  return () => clearTimeout(res);
};

// const result = [];
//
// const fn1 = (x) => x * 5;
// const args = [2],
//   t = 20,
//   cancelTimeMs = 50;
//
// const start = performance.now();
//
// const log = (...argsArr) => {
//   const diff = Math.floor(performance.now() - start);
//   result.push({ time: diff, returned: fn1(...argsArr) });
// };
//
// const cancel = cancellable(log, args, t);
//
// const maxT = Math.max(t, cancelTimeMs);
//
// setTimeout(cancel, cancelTimeMs);
//
// setTimeout(() => {
//   console.log(result); // [{ time: 20, returned: 10 }];
// }, maxT + 15);
