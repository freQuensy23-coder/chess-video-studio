# Renderer artwork

`face.svg`, `face-female.svg`, and `pieces.svg` are original vector artwork created for Chess Video Studio and distributed under the repository's GPL-3.0 license. Both presenters are fictional adult illustrated characters, not reconstructions of real people.

Timelines select `presenter: "male"` (the default) or `presenter: "female"` independently of player color and board orientation. `write_showcase(..., presenter="female")` uses the female presenter throughout its scenarios. Both assets share a 600 × 480 frame and facial rig: eyes at (264, 224) and (338, 224), eyebrows around y=200, and mouth around y=285. The renderer supplies expressions, blinks, audio-driven mouth movement, and reaction zoom; the SVGs supply the distinct hair, face shape, and clothing.

No reference video, reference frames, third-party piece sprites, or identity assets are included.
