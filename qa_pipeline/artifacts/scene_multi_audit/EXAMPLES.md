# Scene multi-target annotation audit examples

Boxes and captions below are read directly from each original `meta_info.json`; A–E follow the original `ground_info` order.

## N1: 1 target(s)

- Split: `train`
- Scene: `waymo/1024360143612057520_3580_000_3600_000/0025_2`
- Local annotated image: `D:\大三上\asc\3eedQA\grounding_QA_publish_http1\qa_pipeline\artifacts\scene_multi_audit\n1_annotated.jpg`

### Object A (pedestrian)

The pedestrian, wearing a dark long coat and black pants, stands on the left middle side of the view, facing the observer from the side with the right hand raised near the head. There is a silver car in front on the right and an orange guardrail behind.

- 2D box: `[335, 706, 649, 1157]`
- 3D box: `[9.387263995267858, -4.992391257410418, 0.8298255941370485, 1.0875911001198517, 0.8059965415287108, 1.8100000000000023, 1.529687026729409]`

## N2: 2 target(s)

- Split: `train`
- Scene: `waymo/1024360143612057520_3580_000_3600_000/0030_2`
- Local annotated image: `D:\大三上\asc\3eedQA\grounding_QA_publish_http1\qa_pipeline\artifacts\scene_multi_audit\n2_annotated.jpg`

### Object A (pedestrian)

The pedestrian, wearing a dark long coat and black pants with white sports shoes, is located in the lower left corner of the view, facing the observer from the side, showing a walking posture. A silver sedan is behind and to the right, and an orange guardrail is behind and to the left.

- 2D box: `[178, 703, 511, 1180]`
- 3D box: `[9.423861766968912, -4.302116905528237, 0.8534956500834596, 1.0875911001198517, 0.8059965415239315, 1.8100000000000023, 1.5419469481966166]`

### Object B (pedestrian)

The pedestrian, wearing a dark top and light-colored pants with a backpack, is located in the lower right corner of the view with part of the body obscured. The side of the body faces the observer, with a silver car on the left and a white fire hydrant in front to the right.

- 2D box: `[1715, 694, 1850, 914]`
- 3D box: `[8.445422852360934, -16.141064146022472, 0.8217545918305404, 0.898450790627976, 0.7202960169574072, 1.6500000000000057, -1.5044819641093476]`

## N3: 3 target(s)

- Split: `train`
- Scene: `waymo/1024360143612057520_3580_000_3600_000/0125_3`
- Local annotated image: `D:\大三上\asc\3eedQA\grounding_QA_publish_http1\qa_pipeline\artifacts\scene_multi_audit\n3_annotated.jpg`

### Object A (pedestrian)

Object type: pedestrian; Main appearance features: wearing a gray coat and black pants, carrying a gray backpack; Location information: slightly to the right of the center in the view, close to the edge of the sidewalk; Relationship with the surrounding environment: walking away from the observer, with a pedestrian on the left and a black sedan on the right.

- 2D box: `[712, 246, 932, 726]`
- 3D box: `[0.9779000258859014, 8.463410377800756, 0.9903422687052625, 1.0875911001198517, 0.805996541526676, 1.8100000000000023, 1.546555323580864]`

### Object B (pedestrian)

Object type: pedestrian; Main appearance features: wearing a blue top and light-colored pants, carrying a backpack, and wearing a hat; Position information: located on the left middle side of the view, with part of the body being obscured; Relationship with the surrounding environment: there is a person wearing a gray coat in front on the right, and a red mailbox on the left.

- 2D box: `[419, 239, 707, 638]`
- 3D box: `[-0.32895270198059734, 9.936910775488286, 1.0074829301277077, 1.178868433489246, 1.180890156172691, 1.7687841945289051, -1.5433864165059115]`

### Object C (car)

Dark-colored sedan with a logo on the front, located slightly to the right of center in the view, facing the observer. There is a white car in front and a pedestrian on the left.

- 2D box: `[1210, 280, 1716, 595]`
- 3D box: `[4.572818882872525, 13.528360847958538, 0.7037153736611117, 4.750193399124475, 2.0782447080498327, 1.509999999999991, -1.5638002487348275]`

## N4: 4 target(s)

- Split: `train`
- Scene: `waymo/10448102132863604198_472_000_492_000/0154_1`
- Local annotated image: `D:\大三上\asc\3eedQA\grounding_QA_publish_http1\qa_pipeline\artifacts\scene_multi_audit\n4_annotated.jpg`

### Object A (car)

This is a large silver SUV with closed windows, located slightly left of center in the view, partially obscured by a tree trunk. It faces the observer from its side, with the front facing the right side of the image. There is a red SUV in front and to the left, and a white SUV behind and to the right.

- 2D box: `[448, 671, 987, 901]`
- 3D box: `[14.92441730413384, 16.434372909033286, 0.482907321339586, 5.187611679281736, 2.3337949996672207, 1.8900000000000006, -3.1098638219246357]`

### Object B (car)

This is a blue SUV located slightly to the right of the center of the view, close to the edge of the road, side facing the observer, front facing left. There is a silver SUV on the left and a white SUV on the right.

- 2D box: `[913, 684, 1257, 824]`
- 3D box: `[20.705456118624284, 16.5614355838773, 0.44909855726710735, 4.329144390923663, 2.0017242958086374, 1.3999999999999986, -3.072384208599835]`

### Object C (car)

This is a white SUV with smooth body lines, located slightly to the right of center and close to the right edge of the view. The side faces the observer, and the front faces left. There is a black SUV on its right and a silver SUV on its left.

- 2D box: `[1217, 676, 1529, 808]`
- 3D box: `[26.815621875049715, 16.435742016285026, 0.32431144257247535, 4.4756975465305695, 2.184168976292381, 1.6099999999999994, 3.091890783611002]`

### Object D (car)

Black SUV, side facing the observer, located slightly to the right in the view, with a white SUV on its left and a dark-colored SUV on its right.

- 2D box: `[1495, 680, 1801, 800]`
- 3D box: `[33.800260592974155, 15.657289039956778, 0.07540006250477216, 5.289360398845904, 2.14661929694849, 1.6900000000000004, 2.988654820912493]`

## N5: 5 target(s)

- Split: `val`
- Scene: `waymo/12940710315541930162_2660_000_2680_000/0049_3`
- Local annotated image: `D:\大三上\asc\3eedQA\grounding_QA_publish_http1\qa_pipeline\artifacts\scene_multi_audit\n5_annotated.jpg`

### Object A (car)

A small silver car with a streamlined design, located slightly left of center in the view, facing the observer, with a dark gray car on its left and a red car on its right.

- 2D box: `[415, 229, 686, 401]`
- 3D box: `[-2.4830692653390543, 21.486092555476716, 1.0091170074379079, 4.651116903012759, 2.1300373946724824, 1.5, -1.5153760449625644]`

### Object B (car)

A black BMW sedan is located slightly left of center in the view, with its side facing the observer and the front pointing towards the right side of the image. There is a white sedan on the left and a red sedan on the right.

- 2D box: `[675, 247, 904, 414]`
- 3D box: `[-0.06951376879078452, 21.779595527914353, 0.8227921523605346, 4.54492367040816, 2.1117401396671815, 1.4700000000000273, -1.5561224240779319]`

### Object C (car)

The red sedan is located slightly to the right of the center of the view, facing the observer directly, with a gray sedan on its left and an orange sedan on its right.

- 2D box: `[942, 247, 1173, 414]`
- 3D box: `[2.495059808004953, 21.902524090754014, 0.8115650474878748, 4.606178893828595, 2.184144146230605, 1.4700000000000273, -1.5681037626665155]`

### Object D (car)

A small orange car with a compact body is located on the right middle side, close to the right edge of the view, with its side facing the observer and the front pointing towards the left side of the image. There is a red car on the left and a black SUV on the right.

- 2D box: `[1174, 237, 1454, 399]`
- 3D box: `[5.201250997438365, 21.824919526725353, 0.9344604476241329, 4.0139568272700625, 2.0469242330913833, 1.4700000000000273, -1.5162077249675963]`

### Object E (car)

This is a small black car with the front part visible, located slightly to the left and close to the left edge of the view, partially obscured. There is a silver car on the right and a blue car on the left.

- 2D box: `[138, 249, 468, 410]`
- 3D box: `[-5.129082227822892, 21.92122376563566, 0.8303147698948123, 4.681782195992031, 2.021995680101796, 1.3999999999999773, -1.5623503988022978]`
