module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module' },
  plugins: ['@typescript-eslint', 'react-hooks'],
  rules: {
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',
    '@typescript-eslint/no-explicit-any': 'error',
    // The frontend must never re-derive a trading value; forbid the arithmetic
    // that would let it drift from the Python engines.
    'no-restricted-syntax': ['error', {
      selector: "BinaryExpression[operator='>'][left.name='prediction']",
      message: 'Signal direction comes from the API, never from a comparison here.',
    }],
  },
  ignorePatterns: ['dist', 'node_modules', '*.cjs'],
};
