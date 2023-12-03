import type { MarkdownInstance } from 'astro-boilerplate-components';

import type { IProjectFormat } from './Interfaces';

export const sortByDate = (posts: MarkdownInstance<IProjectFormat>[]) => {
  return posts.sort(
    (a, b) =>
      new Date(b.frontmatter.pubDate).valueOf() -
      new Date(a.frontmatter.pubDate).valueOf()
  );
};
