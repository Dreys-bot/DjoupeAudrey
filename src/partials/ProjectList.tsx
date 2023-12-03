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

const ProjectList = (props: IRecentProjectsProps) => (
  <Section
    title={
      <>
        Recent <GradientText>Projects</GradientText>
      </>
    }
  >
    <div className="flex flex-col gap-6">
      {props.projectList.map((project) => (
        <Project
          name={project.frontmatter.title}
          description={project.frontmatter.description}
          link="/"
          img={{
            src: project.frontmatter.imgSrc,
            alt: project.frontmatter.imgAlt,
          }}
          category={
            <>
              {project.frontmatter.tags?.map((tag, index) => (
                <Tags color={Object.keys(ColorTags)[(index + 5) % 22]}>{tag}</Tags>
              ))}

              {/* <Tags color={ColorTags.LIME}>Python</Tags>
              <Tags color={ColorTags.SKY}>Google Colab</Tags>
              <Tags color={ColorTags.ROSE}>AWS</Tags> */}
            </>
          }
        />
      ))}
      {/* <Project
        name="Project 2"
        description="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse
        bibendum. Nunc non posuere consectetur, justo erat semper enim, non
        hendrerit dui odio id enim."
        link="/"
        img={{ src: '/assets/images/project-fire.png', alt: 'Project Fire' }}
        category={
          <>
            <Tags color={ColorTags.VIOLET}>Tensorflow</Tags>
            <Tags color={ColorTags.EMERALD}>Python</Tags>
            <Tags color={ColorTags.YELLOW}>Google Colab</Tags>
          </>
        }
      />
      <Project
        name="Project 3"
        description="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse
        bibendum. Nunc non posuere consectetur, justo erat semper enim, non
        hendrerit dui odio id enim."
        link="/"
        img={{ src: '/assets/images/project-maps.png', alt: 'Project Maps' }}
        category={
          <>
            <Tags color={ColorTags.FUCHSIA}>Tensorflow</Tags>
            <Tags color={ColorTags.INDIGO}>Python</Tags>
            <Tags color={ColorTags.ROSE}>Google Colab</Tags>
          </>
        }
      /> */}
    </div>
  </Section>
);

export { ProjectList };
