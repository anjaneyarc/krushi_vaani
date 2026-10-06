import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';

class WeatherPage extends StatefulWidget {
  const WeatherPage({super.key});

  @override
  State<WeatherPage> createState() => _WeatherPageState();
}

class _WeatherPageState extends State<WeatherPage> {
  bool isLoading = true;
  String errorMessage = '';

  double temperature = 0;
  int humidity = 0;
  double windSpeed = 0;
  double rain = 0;
  int weatherCode = 0;

  @override
  void initState() {
    super.initState();
    fetchWeather();
  }

  Future<void> fetchWeather() async {
    try {
            bool serviceEnabled = await Geolocator.isLocationServiceEnabled();

      if (!serviceEnabled) {
        setState(() {
          errorMessage = 'GPS/Location ON ಮಾಡಿ.';
          isLoading = false;
        });
        return;
      }

      LocationPermission permission = await Geolocator.checkPermission();

      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        setState(() {
          errorMessage = 'Location permission ನೀಡಬೇಕು.';
          isLoading = false;
        });
        return;
      }

      Position position = await Geolocator.getCurrentPosition();

      final latitude = position.latitude;
      final longitude = position.longitude;
      

      final url = Uri.parse(
        'https://api.open-meteo.com/v1/forecast'
        '?latitude=$latitude'
        '&longitude=$longitude'
        '&current=temperature_2m,relative_humidity_2m,'
        'precipitation,weather_code,wind_speed_10m'
        '&timezone=auto',
      );

      final response = await http.get(url);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final current = data['current'];

        setState(() {
          temperature = (current['temperature_2m'] ?? 0).toDouble();
          humidity = (current['relative_humidity_2m'] ?? 0).toInt();
          windSpeed = (current['wind_speed_10m'] ?? 0).toDouble();
          rain = (current['precipitation'] ?? 0).toDouble();
          weatherCode = (current['weather_code'] ?? 0).toInt();

          isLoading = false;
        });
      } else {
        setState(() {
          errorMessage = 'ಹವಾಮಾನ ಮಾಹಿತಿ ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.';
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Internet connection check ಮಾಡಿ.';
        isLoading = false;
      });
    }
  }

  String getWeatherDescription() {
    if (weatherCode == 0) {
      return 'ಸ್ವಚ್ಛ ಆಕಾಶ ☀️';
    } else if (weatherCode <= 3) {
      return 'ಮೋಡ ಕವಿದ ವಾತಾವರಣ ☁️';
    } else if (weatherCode >= 51 && weatherCode <= 67) {
      return 'ಮಳೆಯ ಸಾಧ್ಯತೆ 🌧️';
    } else if (weatherCode >= 80 && weatherCode <= 82) {
      return 'ಮಳೆ 🌧️';
    } else if (weatherCode >= 95) {
      return 'ಗುಡುಗು ಸಹಿತ ಮಳೆ ⛈️';
    }

    return 'ಸಾಮಾನ್ಯ ಹವಾಮಾನ';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'ಹವಾಮಾನ ಮಾಹಿತಿ',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: isLoading
          ? const Center(
              child: CircularProgressIndicator(),
            )
          : errorMessage.isNotEmpty
              ? Center(
                  child: Text(
                    errorMessage,
                    style: const TextStyle(fontSize: 17),
                  ),
                )
              : RefreshIndicator(
                  onRefresh: fetchWeather,
                  child: ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      Card(
                        elevation: 4,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(20),
                          child: Column(
                            children: [
                              const Icon(
                                Icons.wb_sunny,
                                size: 70,
                                color: Colors.orange,
                              ),
                              const SizedBox(height: 12),
                              const Text(
                                'ಇಂದಿನ ಹವಾಮಾನ',
                                style: TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 8),
                              const Text(
                                'Mangaluru',
                                style: TextStyle(
                                  fontSize: 17,
                                  color: Colors.grey,
                                ),
                              ),
                              const SizedBox(height: 20),
                              Text(
                                '${temperature.toStringAsFixed(1)}°C',
                                style: const TextStyle(
                                  fontSize: 48,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                getWeatherDescription(),
                                style: const TextStyle(fontSize: 16),
                              ),
                            ],
                          ),
                        ),
                      ),

                      const SizedBox(height: 20),

                      Row(
                        children: [
                          Expanded(
                            child: _weatherCard(
                              Icons.water_drop,
                              'ಆರ್ದ್ರತೆ',
                              '$humidity%',
                              Colors.blue,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: _weatherCard(
                              Icons.air,
                              'ಗಾಳಿ',
                              '${windSpeed.toStringAsFixed(1)} km/h',
                              Colors.teal,
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 12),

                      Row(
                        children: [
                          Expanded(
                            child: _weatherCard(
                              Icons.umbrella,
                              'ಮಳೆ',
                              '${rain.toStringAsFixed(1)} mm',
                              Colors.indigo,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: _weatherCard(
                              Icons.cloud,
                              'ಸ್ಥಿತಿ',
                              'Live',
                              Colors.green,
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 24),

                      Card(
                        elevation: 3,
                        child: Padding(
                          padding: const EdgeInsets.all(18),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                '🌾 ರೈತರಿಗೆ ಸಲಹೆ',
                                style: TextStyle(
                                  fontSize: 19,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 10),
                              Text(
                                rain > 0
                                    ? 'ಮಳೆಯಾಗುತ್ತಿರುವ ಸಾಧ್ಯತೆ ಇದೆ. ಅಗತ್ಯವಿದ್ದರೆ ನೀರಾವರಿ ಕಾರ್ಯವನ್ನು ಮುಂದೂಡಿ ಮತ್ತು ಬೆಳೆಗಳನ್ನು ಗಮನಿಸಿ.'
                                    : 'ಈ ಸಮಯದಲ್ಲಿ ಗಮನಾರ್ಹ ಮಳೆ ಇಲ್ಲ. ನಿಮ್ಮ ಬೆಳೆಗೆ ಅಗತ್ಯವಿರುವ ನೀರಾವರಿ ಮತ್ತು ನಿರ್ವಹಣೆಯನ್ನು ಮುಂದುವರಿಸಿ.',
                                style: const TextStyle(
                                  fontSize: 15,
                                  height: 1.5,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),

                      const SizedBox(height: 16),

                      const Center(
                        child: Text(
                          'Live weather data',
                          style: TextStyle(
                            color: Colors.grey,
                            fontSize: 13,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }

  Widget _weatherCard(
    IconData icon,
    String title,
    String value,
    Color color,
  ) {
    return Card(
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Icon(
              icon,
              size: 32,
              color: color,
            ),
            const SizedBox(height: 8),
            Text(
              title,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 5),
            Text(
              value,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 17,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }
}