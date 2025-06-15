            # Add date filtering to query
            if date_filter == "1h":
                search_query += " after:1h"
            elif date_filter == "6h":
                search_query += " after:6h"
            elif date_filter == "24h":
                search_query += " after:24h"
            elif date_filter == "week":  
                search_query += " after:7d"
            elif date_filter == "month":
                search_query += " after:30d"
            else:
                # Default to last 24 hours for most recent news
                search_query += " after:24h"