import Lenis from 'lenis';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import SplitType from 'split-type';

gsap.registerPlugin(ScrollTrigger);

export function initMotion() {
  const lenis = new Lenis({
    duration: 1.15,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    smoothWheel: true,
    smoothTouch: false,
  });
  window.__lenis = lenis;

  lenis.on('scroll', ScrollTrigger.update);
  gsap.ticker.add((time) => lenis.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);

  // Hero scroll-driven canvas & text animation
  const hero = document.querySelector('#hero');
  const heroContent = document.querySelector('#hero .hero-content');

  if (hero) {
    ScrollTrigger.create({
      trigger: hero,
      start: 'top top',
      end: 'bottom bottom',
      scrub: true,
      onUpdate: (self) => {
        const progress = self.progress;
        if (window.__drawHeroFrame) {
          window.__drawHeroFrame(progress);
        }
      }
    });
  }

  if (hero && heroContent) {
    gsap.timeline({
      scrollTrigger: {
        trigger: hero,
        start: 'top top',
        end: 'bottom bottom',
        scrub: 1.2,
      }
    })
    .to(heroContent, {
      y: -80,
      opacity: 0,
      ease: 'power2.in',
    }, 0);
  }

  // FrankaKitchen scroll animations
  const fk = document.querySelector('#franka-kitchen');

  if (fk && hero) {
    // Transition: hero fades out while FrankaKitchen slides up from below
    gsap.timeline({
      scrollTrigger: {
        trigger: hero,
        start: 'bottom bottom',
        end: 'bottom top',
        scrub: 2,
      }
    })
    .to(hero, { opacity: 0, ease: 'power1.inOut' }, 0)
    .to(fk, { marginTop: '0vh', opacity: 1, ease: 'power1.inOut' }, 0);
  }

  // Section heading reveals
  document.querySelectorAll('section.act h2, section.act h1').forEach((el) => {
    const split = new SplitType(el, { types: 'words,chars' });
    gsap.from(split.chars, {
      y: 40,
      opacity: 0,
      duration: 0.9,
      stagger: 0.015,
      ease: 'power2.out',
      scrollTrigger: {
        trigger: el,
        start: 'top 80%',
        toggleActions: 'play none none none',
      }
    });
  });

  // Section body reveals
  document.querySelectorAll('section.act p, section.act li, section.act .tech-detail-link').forEach((el, i) => {
    gsap.from(el, {
      y: 24,
      opacity: 0,
      duration: 0.7,
      delay: (i % 6) * 0.04,
      ease: 'power2.out',
      scrollTrigger: {
        trigger: el,
        start: 'top 85%',
        toggleActions: 'play none none none',
      }
    });
  });

  return () => {
    lenis.destroy();
    ScrollTrigger.getAll().forEach(t => t.kill());
  };
}
