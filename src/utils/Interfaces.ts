interface IProjectFormat {
  title: string;
  description: string;
  pubDate: string;
  githubPage?: string;
  imgSrc?: string;
  imgAlt?: string;
  iconSrc?: string;
  iconAlt?: string;
  tags?: string[];
}

export type { IProjectFormat };
