from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import json
import requests
from datetime import datetime, timedelta


class TravelResearchInput(BaseModel):
    """Input schema for TravelResearchTool."""
    location: str = Field(..., description="The destination location to research")
    research_type: str = Field(..., description="Type of research: 'attractions', 'food', 'accommodation', 'activities', 'budget'")

class TravelResearchTool(BaseTool):
    name: str = "travel_research_tool"
    description: str = (
        "A comprehensive tool for researching travel destinations. Can research attractions, "
        "food, accommodation, activities, and budget information for any location. "
        "Use this to gather detailed information about travel destinations."
    )
    args_schema: Type[BaseModel] = TravelResearchInput

    def _run(self, location: str, research_type: str) -> str:
        """Research travel information for a specific location and type."""
        try:
            # Simulate research based on type
            if research_type == "attractions":
                return self._research_attractions(location)
            elif research_type == "food":
                return self._research_food(location)
            elif research_type == "accommodation":
                return self._research_accommodation(location)
            elif research_type == "activities":
                return self._research_activities(location)
            elif research_type == "budget":
                return self._research_budget(location)
            else:
                return f"Unknown research type: {research_type}. Available types: attractions, food, accommodation, activities, budget"
        except Exception as e:
            return f"Error researching {location}: {str(e)}"

    def _research_attractions(self, location: str) -> str:
        """Research popular attractions for a location."""
        # This would typically call a real API or database
        attractions_data = {
            "location": location,
            "attractions": [
                {
                    "name": f"Historic Center of {location}",
                    "type": "Cultural Site",
                    "description": "A UNESCO World Heritage site with rich historical significance",
                    "best_time_to_visit": "Morning or late afternoon",
                    "entry_fee": "Varies by attraction",
                    "duration": "2-3 hours"
                },
                {
                    "name": f"{location} Museum",
                    "type": "Museum",
                    "description": "A comprehensive museum showcasing local history and culture",
                    "best_time_to_visit": "Weekday mornings",
                    "entry_fee": "€10-15",
                    "duration": "1-2 hours"
                },
                {
                    "name": f"{location} Park",
                    "type": "Natural Attraction",
                    "description": "Beautiful park perfect for relaxation and outdoor activities",
                    "best_time_to_visit": "Any time during daylight",
                    "entry_fee": "Free",
                    "duration": "1-3 hours"
                }
            ],
            "tips": [
                "Book tickets in advance for popular attractions",
                "Consider getting a city pass for multiple attractions",
                "Check opening hours as they may vary by season"
            ]
        }
        return json.dumps(attractions_data, indent=2)

    def _research_food(self, location: str) -> str:
        """Research food and dining options for a location."""
        food_data = {
            "location": location,
            "local_specialties": [
                f"Traditional {location} dish",
                f"Local street food specialty",
                f"Regional dessert from {location}"
            ],
            "restaurant_recommendations": [
                {
                    "name": f"Best Local Restaurant in {location}",
                    "cuisine": "Local",
                    "price_range": "€€",
                    "rating": "4.5/5",
                    "specialty": "Traditional local dishes"
                },
                {
                    "name": f"Fine Dining {location}",
                    "cuisine": "International",
                    "price_range": "€€€",
                    "rating": "4.8/5",
                    "specialty": "Modern fusion cuisine"
                }
            ],
            "street_food": [
                f"Popular street food in {location}",
                f"Local market specialties"
            ],
            "dining_tips": [
                "Try local specialties at family-run restaurants",
                "Visit local markets for authentic food experiences",
                "Make reservations for popular restaurants"
            ]
        }
        return json.dumps(food_data, indent=2)

    def _research_accommodation(self, location: str) -> str:
        """Research accommodation options for a location."""
        accommodation_data = {
            "location": location,
            "hotels": [
                {
                    "name": f"Luxury Hotel {location}",
                    "type": "5-star",
                    "price_range": "€200-400/night",
                    "location": "City center",
                    "amenities": ["Spa", "Restaurant", "Concierge", "Pool"]
                },
                {
                    "name": f"Boutique Hotel {location}",
                    "type": "4-star",
                    "price_range": "€100-200/night",
                    "location": "Historic district",
                    "amenities": ["Breakfast", "WiFi", "Bar"]
                }
            ],
            "budget_options": [
                {
                    "name": f"Hostel {location}",
                    "type": "Hostel",
                    "price_range": "€20-50/night",
                    "location": "Near city center",
                    "amenities": ["Shared kitchen", "Common room", "WiFi"]
                }
            ],
            "unique_stays": [
                f"Historic building converted to hotel in {location}",
                f"Local guesthouse with traditional architecture"
            ],
            "booking_tips": [
                "Book in advance during peak season",
                "Consider location vs. price trade-offs",
                "Read recent reviews before booking"
            ]
        }
        return json.dumps(accommodation_data, indent=2)

    def _research_activities(self, location: str) -> str:
        """Research activities and things to do for a location."""
        activities_data = {
            "location": location,
            "outdoor_activities": [
                f"Walking tour of {location}",
                f"Bike tour around {location}",
                f"Boat tour on local waterway"
            ],
            "cultural_activities": [
                f"Visit local museums in {location}",
                f"Attend cultural events in {location}",
                f"Take cooking class featuring local cuisine"
            ],
            "day_trips": [
                f"Day trip to nearby historic town",
                f"Visit local natural attractions",
                f"Explore surrounding countryside"
            ],
            "seasonal_activities": {
                "spring": f"Spring festivals in {location}",
                "summer": f"Outdoor concerts in {location}",
                "autumn": f"Harvest festivals near {location}",
                "winter": f"Winter markets in {location}"
            },
            "activity_tips": [
                "Book popular activities in advance",
                "Check weather conditions for outdoor activities",
                "Consider group vs. private tour options"
            ]
        }
        return json.dumps(activities_data, indent=2)

    def _research_budget(self, location: str) -> str:
        """Research budget information for a location."""
        budget_data = {
            "location": location,
            "currency": "EUR",  # This would be dynamic based on location
            "budget_tiers": {
                "budget": {
                    "daily_budget": "€30-50",
                    "accommodation": "€15-30/night",
                    "food": "€10-20/day",
                    "activities": "€5-15/day",
                    "transportation": "€5-10/day"
                },
                "mid_range": {
                    "daily_budget": "€80-150",
                    "accommodation": "€50-100/night",
                    "food": "€30-60/day",
                    "activities": "€20-40/day",
                    "transportation": "€10-25/day"
                },
                "luxury": {
                    "daily_budget": "€200+",
                    "accommodation": "€150+/night",
                    "food": "€80+/day",
                    "activities": "€50+/day",
                    "transportation": "€30+/day"
                }
            },
            "cost_breakdown": {
                "accommodation": "30-40% of total budget",
                "food": "25-35% of total budget",
                "activities": "20-30% of total budget",
                "transportation": "10-20% of total budget"
            },
            "money_saving_tips": [
                "Stay in hostels or budget hotels",
                "Eat at local restaurants and street food",
                "Use public transportation",
                "Look for free activities and attractions",
                "Travel during off-peak seasons"
            ],
            "hidden_costs": [
                "Tourist taxes",
                "Service charges",
                "Tips and gratuities",
                "ATM fees",
                "Travel insurance"
            ]
        }
        return json.dumps(budget_data, indent=2)


class WeatherCheckInput(BaseModel):
    """Input schema for WeatherCheckTool."""
    location: str = Field(..., description="The location to check weather for")
    days_ahead: int = Field(default=7, description="Number of days ahead to check weather (1-14)")

class WeatherCheckTool(BaseTool):
    name: str = "weather_check_tool"
    description: str = (
        "Check current weather conditions and forecast for any location. "
        "Useful for planning outdoor activities and packing recommendations."
    )
    args_schema: Type[BaseModel] = WeatherCheckInput

    def _run(self, location: str, days_ahead: int = 7) -> str:
        """Check weather for a location."""
        try:
            # Simulate weather data (in real implementation, this would call a weather API)
            weather_data = {
                "location": location,
                "current_weather": {
                    "temperature": "22°C",
                    "condition": "Partly cloudy",
                    "humidity": "65%",
                    "wind": "10 km/h"
                },
                "forecast": []
            }
            
            # Generate forecast for the specified number of days
            for i in range(min(days_ahead, 14)):
                date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                weather_data["forecast"].append({
                    "date": date,
                    "high": f"{20 + i}°C",
                    "low": f"{15 + i}°C",
                    "condition": "Sunny" if i % 2 == 0 else "Partly cloudy",
                    "precipitation_chance": f"{10 + i * 5}%"
                })
            
            weather_data["recommendations"] = [
                "Pack layers for varying temperatures",
                "Bring an umbrella for potential rain",
                "Wear comfortable walking shoes",
                "Consider sunscreen for outdoor activities"
            ]
            
            return json.dumps(weather_data, indent=2)
        except Exception as e:
            return f"Error checking weather for {location}: {str(e)}"


class LocalTransportInput(BaseModel):
    """Input schema for LocalTransportTool."""
    location: str = Field(..., description="The location to research transportation for")
    transport_type: str = Field(default="all", description="Type of transport: 'public', 'taxi', 'rental', 'all'")

class LocalTransportTool(BaseTool):
    name: str = "local_transport_tool"
    description: str = (
        "Research local transportation options for any destination. "
        "Provides information about public transport, taxis, car rentals, and other transport methods."
    )
    args_schema: Type[BaseModel] = LocalTransportInput

    def _run(self, location: str, transport_type: str = "all") -> str:
        """Research transportation options for a location."""
        try:
            transport_data = {
                "location": location,
                "public_transport": {
                    "metro": f"Metro system available in {location}",
                    "bus": f"Comprehensive bus network in {location}",
                    "tram": f"Tram system in central {location}",
                    "tickets": "Single tickets, day passes, and multi-day passes available",
                    "cost": "€2-5 per trip, €10-20 for day pass"
                },
                "taxi_services": {
                    "availability": f"Taxis readily available in {location}",
                    "apps": f"Uber, local taxi apps available in {location}",
                    "cost": "€15-30 for short trips, €30-60 for longer trips",
                    "tips": "Use official taxi stands or apps for safety"
                },
                "car_rental": {
                    "companies": f"Major car rental companies in {location}",
                    "requirements": "Valid driver's license, credit card, minimum age 21",
                    "cost": "€30-80 per day depending on car type",
                    "tips": "Book in advance for better rates"
                },
                "other_options": [
                    f"Bike sharing in {location}",
                    f"Walking tours of {location}",
                    f"Boat services in {location}"
                ],
                "transportation_tips": [
                    "Get a travel card for public transport",
                    "Download local transport apps",
                    "Consider walking for short distances",
                    "Check for tourist transport passes"
                ]
            }
            return json.dumps(transport_data, indent=2)
        except Exception as e:
            return f"Error researching transport for {location}: {str(e)}"
