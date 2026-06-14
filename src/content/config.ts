import { defineCollection, z } from 'astro:content';

const notes = defineCollection({
  type: 'content',
  schema: z.object({
    title:       z.string(),
    date:        z.coerce.date(),       // accepts "YYYY-MM-DD" strings
    description: z.string(),
    tags:        z.array(z.string()).optional().default([]),
    draft:       z.boolean().optional().default(false),  // set true to hide
  }),
});

export const collections = { notes };
