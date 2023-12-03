import { Section } from 'astro-boilerplate-components';

import { AppConfig } from '@/utils/AppConfig';

const Footer = () => (
  <Section>
    <div class="border-t border-gray-600 pt-5">
      <div class="text-sm text-gray-200">
        &copy; Copyright {new Date().getFullYear()} by {AppConfig.author}.
      </div>
    </div>
  </Section>
);

export { Footer };
