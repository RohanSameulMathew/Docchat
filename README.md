# LLM document chatter: Work in progress

**Description:** 
This allows you to get relevant chunks out of a document and use an llm to process and respond using that information. This can process pdfs, images, txt files, html files, and websites. It also allows for text to speech output.

 [![Tests](https://github.com/RohanSameulMathew/docchat/actions/workflows/main.yml/badge.svg)](https://github.com/RohanSameulMathew/docchat/actions/workflows/main.yml)

## Good example

```bash
docchat> https://cdn-s3.touchofmodern.com/products/000/631/530/814f5930b2a7ae0c2c2188f011bcd564_large.jpg why is this image funny
Loaded new file/URL from chat!
result= "Shiver me timbers! The image be funny because Spiderman’s thought bubble shows he prefers swingin’ from rooftops over waitin’ for the bus, pokin’ fun at the everyday grind o’ public transit and highlightin’ his super‑powered impatience—savvy?"
```
```bash
docchat> https://cdn-s3.touchofmodern.com/products/000/631/530/814f5930b2a7ae0c2c2188f011bcd564_large.jpg why is the joke funny
Loaded new file/URL from chat!
result= "Shiver me timbers! The joke be funny because it’s a clever play on the common frustration programmers face with errors in their code. The mortal uses wordplay to avoid future bugs, without explicitly statin’ the new rule. Arrr, it be a swashbucklin’ good joke, matey!"
```

In the second run, the assistant mistakenly recycled its prior explanation about a programmer‑error joke rather than addressing the Spiderman comic strip, showing how context can get mis‑applied.


