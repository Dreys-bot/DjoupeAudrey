import {
  GradientText,
  HeroAvatar,
  HeroSocial,
  Section,
} from 'astro-boilerplate-components';

import { SocialMedia } from '@/utils/AppConfig';

const Hero = () => (
  <Section>
    <HeroAvatar
      title={
        <>
          Welcome to my <GradientText>portfolio</GradientText> 👋
        </>
      }
      description={
        <>
          As a data scientist with 2 years of experience at{' '}
          <a
            className="text-cyan-400 hover:underline"
            href="https://www.capgemini.com/"
            target="_blank"
          >
            Capgemini
          </a>
          , I share my knowledge and experiences related to machine learning and
          computer vision. You will find data analyses, tutorials and insights
          from my projects applying AI techniques.
        </>
      }
      // avatar={
      //   <img className="h-80 w-64" src="" alt="Avatar image" loading="lazy" />
      // }
      socialButtons={
        <>
          <a href={SocialMedia.github}>
            <HeroSocial src="/assets/icons/github-icon.png" alt="Github icon" />
          </a>
          <a href={SocialMedia.linkedin}>
            <HeroSocial
              src="/assets/icons/linkedin-icon.png"
              alt="Linkedin icon"
            />
          </a>
        </>
      }
    />
  </Section>
);

export { Hero };
