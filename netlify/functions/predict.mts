import Anthropic from '@anthropic-ai/sdk'
import type { Config } from '@netlify/functions'

const anthropic = new Anthropic()

export default async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 204 })
  }

  let review: string
  try {
    const body = await req.json()
    review = (body.review || '').trim()
  } catch {
    return Response.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  if (!review) {
    return Response.json({ error: 'Review text is required' }, { status: 400 })
  }

  const message = await anthropic.messages.create({
    model: 'claude-haiku-4-5',
    max_tokens: 512,
    messages: [
      {
        role: 'user',
        content: `Analyze this product review and determine if it is FAKE (spam/incentivized/misleading) or REAL (genuine customer experience).

Review:
"""
${review}
"""

Respond ONLY with a JSON object in this exact format, no markdown, no extra text:
{"label":"FAKE","confidence":0.87,"reason":"Brief 1-sentence explanation"}

Rules:
- label must be exactly "FAKE" or "REAL"
- confidence is a number from 0.5 to 1.0 (how certain you are)
- reason is a single short sentence`,
      },
    ],
  })

  const raw = message.content[0].type === 'text' ? message.content[0].text.trim() : ''

  let parsed: { label: string; confidence: number; reason: string }
  try {
    parsed = JSON.parse(raw)
  } catch {
    const labelMatch = raw.match(/"label"\s*:\s*"(FAKE|REAL)"/i)
    const confMatch = raw.match(/"confidence"\s*:\s*([0-9.]+)/)
    const reasonMatch = raw.match(/"reason"\s*:\s*"([^"]+)"/)
    parsed = {
      label: labelMatch ? labelMatch[1].toUpperCase() : 'REAL',
      confidence: confMatch ? parseFloat(confMatch[1]) : 0.5,
      reason: reasonMatch ? reasonMatch[1] : 'Unable to determine',
    }
  }

  return Response.json({
    label: parsed.label === 'FAKE' ? 'FAKE' : 'REAL',
    confidence: Math.min(1, Math.max(0.5, Number(parsed.confidence) || 0.5)),
    reason: String(parsed.reason || ''),
  })
}

export const config: Config = {
  path: '/api/predict',
  method: ['POST'],
}
