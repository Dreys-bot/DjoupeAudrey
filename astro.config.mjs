import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import compress from 'astro-compress';
import robotsTxt from 'astro-robots-txt';
import { astroImageTools } from 'astro-imagetools';

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';

// https://astro.build/config
export default defineConfig({
  // base: '/',
  site: 'https://dreys-bot.github.io/',
  trailingSlash: 'always', // Use to always append '/' at end of url
  integrations: [
    react(),
    tailwind({}),
    sitemap(),
    robotsTxt(),
    compress({
      html: {
        collapseWhitespace: true,
        collapseInlineTagWhitespace: false,
        conservativeCollapse: true,
        minifyCSS: true,
        minifyJS: true,
        minifyURLs: true,
        sortAttributes: true,
        sortClassName: true,
        removeComments: true,
      },
    }),
    astroImageTools,
  ],
  markdown: {
    extendDefaultPlugins: true,
    shikiConfig: {
      theme: 'monokai',
    },
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeRaw],
  },
});
