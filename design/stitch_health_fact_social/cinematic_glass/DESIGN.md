---
name: Cinematic Glass
colors:
  surface: '#101415'
  surface-dim: '#101415'
  surface-bright: '#363a3b'
  surface-container-lowest: '#0b0f10'
  surface-container-low: '#191c1e'
  surface-container: '#1d2022'
  surface-container-high: '#272a2c'
  surface-container-highest: '#323537'
  on-surface: '#e0e3e5'
  on-surface-variant: '#c3c5d9'
  inverse-surface: '#e0e3e5'
  inverse-on-surface: '#2d3133'
  outline: '#8d90a2'
  outline-variant: '#434656'
  surface-tint: '#b7c4ff'
  primary: '#b7c4ff'
  on-primary: '#002682'
  primary-container: '#0052ff'
  on-primary-container: '#dfe3ff'
  inverse-primary: '#004ced'
  secondary: '#b9f1ff'
  on-secondary: '#00363f'
  secondary-container: '#00e0ff'
  on-secondary-container: '#005f6d'
  tertiary: '#c2c6db'
  on-tertiary: '#2b3040'
  tertiary-container: '#616578'
  on-tertiary-container: '#e0e3f9'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#dde1ff'
  primary-fixed-dim: '#b7c4ff'
  on-primary-fixed: '#001452'
  on-primary-fixed-variant: '#0038b6'
  secondary-fixed: '#a5eeff'
  secondary-fixed-dim: '#00daf8'
  on-secondary-fixed: '#001f25'
  on-secondary-fixed-variant: '#004e5a'
  tertiary-fixed: '#dee1f7'
  tertiary-fixed-dim: '#c2c6db'
  on-tertiary-fixed: '#161b2b'
  on-tertiary-fixed-variant: '#414658'
  background: '#101415'
  on-background: '#e0e3e5'
  surface-variant: '#323537'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.04em
  display-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: 0em
  accent-italic:
    fontFamily: Source Serif 4
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: 0.02em
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '700'
    lineHeight: '1'
    letterSpacing: 0.1em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 8px
  container-padding: 40px
  section-gap: 80px
  element-gap: 24px
  glass-padding: 32px
---

## Brand & Style
The design system moves beyond functional healthcare to a high-fidelity, cinematic experience. It targets users who value precision, advanced technology, and a premium aesthetic. The emotional response is one of calm confidence, authority, and "future-state" innovation.

The design style is an evolved **Glassmorphism**. It utilizes multi-layered translucency, depth-focused background blurs, and hyper-refined light-refraction edges. By combining the structural reliability of a medical platform with the dramatic visual hierarchy of a documentary film, the UI feels both clinical and prestigious.

## Colors
The palette is rooted in a deep, nocturnal foundation to enhance the "Cinematic Glass" effect. 

- **Primary (#0052FF):** The core medical blue, used for primary actions and essential data points.
- **Secondary (#00E0FF):** An electric cyan used for "high-tech" accents, active states, and glow effects.
- **Deep Navy (#0A0F1E):** The base background color, providing the necessary contrast for translucent layers.
- **Surface Tints:** Use white with low opacity (5% to 12%) for glass containers to ensure the background imagery or gradients bleed through correctly.

## Typography
Typography is used to create a documentary-style narrative. 

- **Headlines:** Use Plus Jakarta Sans with tight tracking and bold weights for a commanding, cinematic presence.
- **Body:** Inter provides a clean, systematic feel for readability against complex backgrounds.
- **Accents:** Source Serif 4 (Italic) is used for descriptions, captions, or "Director's Note" style annotations, adding a refined, editorial layer to the clinical data.
- **Labels:** Small, all-caps labels with wide tracking are used for technical metadata and category headers to reinforce the "interface of the future" look.

## Layout & Spacing
This design system employs a **Fluid Grid** with generous atmospheric margins. 

- **Breathing Room:** Increase standard vertical padding between sections to 80px or more to maintain a "wide-angle" cinematic feel.
- **Glass Padding:** Internal padding for glass cards should be at least 32px to ensure content doesn't feel cramped against the frosted edges.
- **Mobile Reflow:** On mobile, margins reduce to 20px, and the multi-column desktop layouts collapse into a single-column stack, maintaining the glass depth via vertical layering rather than horizontal proximity.

## Elevation & Depth
Depth is the defining characteristic of this design system.

- **The Background:** Use a "Cinematic Floor"—a deep navy gradient mixed with blurred abstract medical imagery (macroscopic cells or light refractions).
- **Glass Layers:** Containers use a `backdrop-filter: blur(20px)` and a background of `rgba(255, 255, 255, 0.08)`.
- **The "Light Edge":** Every glass element must have a 1px border using `rgba(255, 255, 255, 0.2)`. This simulates light catching the edge of a physical lens.
- **Inner Glow:** Apply a subtle inner shadow (`inset 0 1px 1px rgba(255, 255, 255, 0.1)`) to give the glass a sense of thickness.
- **Shadows:** Avoid pitch-black shadows. Use ultra-diffused, large-radius shadows tinted with the primary blue (`rgba(0, 82, 255, 0.15)`).

## Shapes
Shapes are sophisticated and intentional. While the "Rounded" setting (0.5rem base) provides a modern feel, larger container elements should scale up to `rounded-xl` (1.5rem) to soften the cinematic framing. Interactive elements like buttons should maintain a consistent, confident radius that mirrors the primary glass containers.

## Components
- **Glass Cards:** The primary vessel for data. Must include the 1px border and backdrop blur. Headers within cards should use the `accent-italic` serif for descriptions.
- **Cinematic Buttons:** Primary buttons use a solid `#0052FF` with a subtle outer glow of the same color. Secondary buttons are "Ghost Glass"—transparent with only the 1px white/20 border.
- **Data Visualizations:** Charts should use vibrant, glowing lines (using the Secondary Cyan) against the dark glass background. 
- **Input Fields:** Minimalist design with only a bottom border that illuminates to the Secondary Cyan color when focused.
- **Status Chips:** Use high-saturation colors (Electric Blue, Emerald, or Ruby) but keep them small and pill-shaped so they act as "indicator lights" on the console.
- **Focus States:** Any focused element should gain a 4px "aura" or soft glow using the Secondary color to signify active engagement in the futuristic interface.