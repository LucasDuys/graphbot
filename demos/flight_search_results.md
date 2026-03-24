# Flight Search Demo Results

Generated: 2026-03-24 16:28:47
Mode: dry-run
Delivery: Console

## Task

Find cheap flights from Amsterdam to Barcelona next week and message me the results on WhatsApp.

## DAG Decomposition

```
browser_search -> data_extraction -> formatting -> whatsapp_send
```

| Node | Domain | Description |
|------|--------|-------------|
| `flight_search_browser` | browser | Navigate to https://www.google.com/travel/flights and search... |
| `flight_search_extract` | synthesis | Parse the raw flight search results text into structured dat... |
| `flight_search_format` | synthesis | Format the structured flight data into a clean, readable mes... |
| `flight_search_send` | comms | Send the formatted flight results message via WhatsApp. Fall... |

## Flight Data (structured)

```json
[
  {
    "airline": "Transavia",
    "flight_no": "HV5143",
    "departure": "Mon 30 Mar 06:25",
    "arrival": "Mon 30 Mar 09:00",
    "duration": "2h 35m",
    "stops": "Direct",
    "price_eur": 49
  },
  {
    "airline": "Vueling",
    "flight_no": "VY8408",
    "departure": "Mon 30 Mar 10:15",
    "arrival": "Mon 30 Mar 12:55",
    "duration": "2h 40m",
    "stops": "Direct",
    "price_eur": 67
  },
  {
    "airline": "KLM",
    "flight_no": "KL1677",
    "departure": "Mon 30 Mar 14:30",
    "arrival": "Mon 30 Mar 17:05",
    "duration": "2h 35m",
    "stops": "Direct",
    "price_eur": 112
  },
  {
    "airline": "easyJet",
    "flight_no": "U27802",
    "departure": "Mon 30 Mar 07:40",
    "arrival": "Mon 30 Mar 10:20",
    "duration": "2h 40m",
    "stops": "Direct",
    "price_eur": 54
  },
  {
    "airline": "Ryanair",
    "flight_no": "FR3812",
    "departure": "Mon 30 Mar 19:55",
    "arrival": "Mon 30 Mar 22:30",
    "duration": "2h 35m",
    "stops": "Direct",
    "price_eur": 39
  },
  {
    "airline": "Lufthansa",
    "flight_no": "LH989",
    "departure": "Mon 30 Mar 12:10",
    "arrival": "Mon 30 Mar 16:50",
    "duration": "4h 40m",
    "stops": "1 stop (FRA)",
    "price_eur": 143
  }
]
```

## Formatted Message

```
Cheap flights: Amsterdam (AMS) -> Barcelona (BCN)
Week of 30 Mar 2026

#   Airline      Flight    Departs                Duration  Stops           Price
--------------------------------------------------------------------------------
1   Ryanair      FR3812    Mon 30 Mar 19:55       2h 35m    Direct         EUR  39
2   Transavia    HV5143    Mon 30 Mar 06:25       2h 35m    Direct         EUR  49
3   easyJet      U27802    Mon 30 Mar 07:40       2h 40m    Direct         EUR  54
4   Vueling      VY8408    Mon 30 Mar 10:15       2h 40m    Direct         EUR  67
5   KLM          KL1677    Mon 30 Mar 14:30       2h 35m    Direct         EUR 112
6   Lufthansa    LH989     Mon 30 Mar 12:10       4h 40m    1 stop (FRA)   EUR 143
--------------------------------------------------------------------------------
Best deal: Ryanair FR3812 at EUR 39 (2h 35m, Direct)

Prices are per person, one-way, including taxes.
```
