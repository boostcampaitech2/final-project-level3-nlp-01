module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  extends: [
    'airbnb-base',
  ],
  overrides: [ // this stays the same
    {
      files: ['*.svelte'],
      processor: 'svelte3/svelte3',
    },
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 13,
    sourceType: 'module',
  },
  plugins: [
    'svelte3',
    '@typescript-eslint', // add the TypeScript plugin
  ],
  rules: {
  },
  settings: {
    'svelte3/typescript': true, // load TypeScript as peer dependency
  },
};
