import type { MarkdownInstance } from 'astro';
import {
  ColorTags,
  GradientText,
  Project,
  Section,
  Tags,
} from 'astro-boilerplate-components';

import type { IProjectFormat } from '@/utils/Interfaces';

type IRecentProjectsProps = {
  projectList: MarkdownInstance<IProjectFormat>[];
};

const RecentProjects = ({ projectList }: IRecentProjectsProps) => {
  return (
    <Section
      title={
        <>
          Recent <GradientText>Projects</GradientText>
        </>
      }
    >
      <div className="flex flex-col gap-6">
        {projectList.map((project) => {
          // Project link
          const link = project.frontmatter.githubPage
            ? project.frontmatter.githubPage
            : '/';

          // Icon props
          const iconSrc = project.frontmatter.iconSrc
            ? project.frontmatter.iconSrc
            : '';
          const iconAlt = project.frontmatter.iconAlt
            ? project.frontmatter.iconAlt
            : '';

          return (
            <Project
              name={project.frontmatter.title}
              description={project.frontmatter.description}
              link={link}
              img={{ src: iconSrc, alt: iconAlt }}
              category={
                <>
                  {project.frontmatter.tags?.map((tag, index) => (
                    <Tags color={Object.keys(ColorTags)[(index + 5) % 22]}>
                      {tag}
                    </Tags>
                  ))}
                </>
              }
            />
          );
        })}
      </div>
    </Section>
  );
};

export { RecentProjects };
