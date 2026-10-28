# Placebound figures

Twenty stylised fantasy figures, modelled in Blender for the board game [Placebound](https://placebound.dariuszparys.com). They are dedicated to the public domain under [CC0 1.0](LICENSE): use them for anything, commercial or not, with no attribution required.

Each figure comes as a game-ready `.glb`, an editable `.blend`, and the Python script that builds it.

## The figures

Each preview shows four views (board angle, front, side, back) in the Light palette above the Dark one.

### Light

| | |
| --- | --- |
| **Knight** (troop)<br>![Knight](previews/knight.png) | **Archer** (troop)<br>![Archer](previews/archer.png) |
| **Unicorn** (beast)<br>![Unicorn](previews/unicorn.png) | **Golem** (heavy)<br>![Golem](previews/golem.png) |
| **Valkyrie** (flyer)<br>![Valkyrie](previews/valkyrie.png) | **Djinni** (flyer)<br>![Djinni](previews/djinni.png) |
| **Phoenix** (flyer)<br>![Phoenix](previews/phoenix.png) | **Wizard** (commander)<br>![Wizard](previews/wizard.png) |

### Dark

| | |
| --- | --- |
| **Goblin** (troop)<br>![Goblin](previews/goblin.png) | **Manticore** (troop)<br>![Manticore](previews/manticore.png) |
| **Basilisk** (beast)<br>![Basilisk](previews/basilisk.png) | **Troll** (heavy)<br>![Troll](previews/troll.png) |
| **Banshee** (flyer)<br>![Banshee](previews/banshee.png) | **Dragon** (flyer)<br>![Dragon](previews/dragon.png) |
| **Shapeshifter** (flyer)<br>![Shapeshifter](previews/shapeshifter.png) | **Sorceress** (commander)<br>![Sorceress](previews/sorceress.png) |

### Elementals (either side)

| | |
| --- | --- |
| **Earth** (heavy)<br>![Earth](previews/earth.png) | **Fire** (elemental)<br>![Fire](previews/fire.png) |
| **Air** (flyer)<br>![Air](previews/air.png) | **Water** (elemental)<br>![Water](previews/water.png) |

## What's here

| Path | What it is |
| --- | --- |
| `models/<kind>.glb` | The game-ready model, 15,000–18,000 triangles, 210–520 KB |
| `blend/<Kind>.blend` | The finished figure, to open and edit in Blender |
| `scripts/<kind>.py` | Builds one kind with the helpers in `lib.py`; the script is the source of the figure |
| `scripts/lib.py` | The modelling contract: scale, orientation, material roles, baking, validation, export and previews |
| `scripts/build.py` | Runs one kind's script in a background Blender |
| `scripts/build_all.py` | Rebuilds every kind |
| `previews/<kind>.png` | The preview sheets shown above |

## Use the models

The `.glb` files are glTF 2.0 with these properties:

- **Compressed.** The meshes use `EXT_meshopt_compression`, so your loader must support it. Blender imports them directly. In three.js, give `GLTFLoader` the `MeshoptDecoder`.
- **Coloured by role, not by texture.** There are no textures. Every face uses one of six materials named `body`, `cloth`, `skin`, `trim`, `metal` and `eye`, exported in plain grey. Assign each role a colour yourself, so one model serves any army. The palettes below are the ones in the previews.
- **Shading baked in.** Ambient occlusion, crease darkening and edge highlights are baked into the vertex colours (`COLOR_0`). Multiply them with the role colour.
- **Two-sided.** Cloaks, sleeves and wings are open shapes, so render the materials double-sided.
- **Scale and placement.** One unit is one figure unit; the tallest figures stand about 1.45 high. The figure faces +Z with Y up, and its feet rest on the origin. The models have no base, so add your own. Flyers hover above the origin.

| Role | Light | Dark |
| --- | --- | --- |
| `body` | `#EEF0E4` | `#241E29` |
| `cloth` | `#B4CFCB` | `#3E2C35` |
| `skin` | `#F0DAC6` | `#9E8E98` |
| `trim` | `#35B8AA` | `#FF8A4C`, emissive `#FF6A2A` |
| `metal` | `#C3CCD4` | `#5D525C` |
| `eye` | `#6FE0D2`, emissive `#35B8AA` | `#FFB27A`, emissive `#FFB27A` |

A minimal three.js example:

```js
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const light = {
  body: 0xeef0e4, cloth: 0xb4cfcb, skin: 0xf0dac6,
  trim: 0x35b8aa, metal: 0xc3ccd4, eye: 0x6fe0d2,
};

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const { scene: knight } = await loader.loadAsync("models/knight.glb");
knight.traverse((o) => {
  if (!o.isMesh) return;
  o.material = new THREE.MeshStandardMaterial({
    color: light[o.material.name],
    vertexColors: true,
    side: THREE.DoubleSide,
  });
});
```

## Rebuild a figure

Blender 5.2 or later must be installed. From the repository root:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python scripts/build.py -- Knight --sheet /tmp/knight.png --board /tmp/knight-board.png --with Archer,Goblin
```

The last line printed starts with `REPORT` and gives the triangle count, bounds, the share of each material role and any problems.

- `--sheet` renders four views (board angle, front, side, back) in the Light palette above the Dark one.
- `--board` renders the figure at board scale beside the `--with` kinds, as the game's board camera shows them.
- `--no-save` leaves the `.blend` and `.glb` untouched.

To rebuild every figure, run `python3 scripts/build_all.py`. Add `--previews` to render the preview sheets too. Set `BLENDER` if Blender is not at the macOS default path.

A rebuild is not byte-identical, so rebuilding changes every `.blend` and `.glb` slightly.

## The contract

The scripts check every figure against these rules; `REPORT` lists any it breaks.

- **Scale and placement.** Blender units are figure units. Z is up, the figure faces −Y, and the origin is the top of its plinth. Grounded figures stand on z = 0; flyers hover with their lowest point at 0.15 or above, solid parts above about 0.34.
- **Height by tier.** The top of the figure falls in its tier's range, so the figures keep their size order: troop 0.64–0.74, beast 0.80–0.88, heavy 0.94–1.12, flyer 0.85–1.22, commander 1.32–1.45 (always the tallest), elemental 0.70–0.86.
- **Footprint.** Within ±0.46 in x and ±0.56 in y.
- **Budget.** At most 18,000 triangles.
- **Material roles.** Every face uses one of the six role materials. `body` and `cloth` cover at least 40% of the surface so the side reads at a glance; `trim` marks the side's accent colour; `eye` glows.
- **Shading.** Ambient occlusion, crease darkening and edge highlights are baked into the `AO` colour attribute.

## Art direction

Stylised heroic fantasy, in the spirit of painted tabletop miniatures: readable from far away first, rich up close second.

- **Silhouette first.** A player must name the piece from the board camera (40° above the horizon, about 80 pixels tall). The identifying features are exaggerated: weapons, horns, wings, hats, tails.
- **Heroic proportions.** Slightly large heads, hands, weapons and feet; clear gesture in the pose.
- **Three levels of form.** Big shapes (silhouette), secondary shapes (armour plates, muscle groups, cloth folds), small details (rivets, straps, buckles, scales, feather tips).
- **Crisp hard surfaces, soft organic ones.** Armour and weapons have bevelled edges and visible thickness; flesh, cloth and fur are smooth with defined anatomy.
- **Both palettes.** Every figure reads in ivory and teal (Light) and in charcoal and ember (Dark), with `trim` leading the eye to what identifies the piece.

## License

[CC0 1.0 Universal](LICENSE). To the extent possible under law, the author has waived all copyright and related rights to these models, Blender files, scripts and previews. A link back to this repository is appreciated but not required.
