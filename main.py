    async def write_article_claude(self, verified_facts: str) -> str:
        """Write article using Claude Sonnet 4"""
        try:
            message = await anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Write a professional AP Style news article from these verified facts:

{verified_facts}

Requirements: 300-400 words, lead paragraph with most newsworthy fact, inverted pyramid structure, objective tone, proper attribution."""
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error in Claude article writing: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to write article: {str(e)}")