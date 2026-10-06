import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'weather_page.dart';
import 'farmer_profile_page.dart';
import 'crop_advisory_page.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const KrushiVaaniApp());
}

class KrushiVaaniApp extends StatelessWidget {
  const KrushiVaaniApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'KrushiVaani',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.green,
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor:
            const Color(0xFFF6FBF6),
      ),
      home: const HomePage(),
    );
  }
}

// =========================================================
// HOME PAGE
// =========================================================

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // =========================================================
  // SERVICES
  // =========================================================

  final stt.SpeechToText speech =
      stt.SpeechToText();

  final FlutterTts flutterTts =
      FlutterTts();

  final ImagePicker imagePicker =
      ImagePicker();

  // =========================================================
  // STATE
  // =========================================================

  bool isListening = false;
  bool isAnalyzingImage = false;

  XFile? selectedImage;

  String recognizedText = '';
  String backendAnswer = '';

  List<String> mainAdvice = [];
  List<String> doSteps = [];
  List<String> dontSteps = [];

  String nextStep = '';

  String detectedCrop = '';
  String detectedDisease = '';

  double confidence = 0.0;

  Map<String, dynamic>? diseaseInfo;

  // =========================================================
  // BACKEND
  // =========================================================

  static const String backendUrl =
      'http://192.168.1.39:8000';

  // =========================================================
  // INITIALIZATION
  // =========================================================

  @override
  void initState() {
    super.initState();
    initializeTts();
  }

  Future<void> initializeTts() async {
    try {
      await flutterTts.setLanguage('kn-IN');
      await flutterTts.setSpeechRate(0.45);
      await flutterTts.setVolume(1.0);
      await flutterTts.setPitch(1.0);
      await flutterTts.awaitSpeakCompletion(true);
    } catch (e) {
      debugPrint(
        'TTS Initialization Error: $e',
      );
    }
  }

  @override
  void dispose() {
    speech.stop();
    flutterTts.stop();
    super.dispose();
  }

  // =========================================================
  // TTS
  // =========================================================

  Future<void> speakAnswer(String text) async {
    if (text.trim().isEmpty) {
      return;
    }

    try {
      await flutterTts.stop();

      await flutterTts.setLanguage('kn-IN');
      await flutterTts.setSpeechRate(0.45);
      await flutterTts.setVolume(1.0);
      await flutterTts.setPitch(1.0);

      await flutterTts.speak(text);
    } catch (e) {
      debugPrint(
        'TTS Error: $e',
      );
    }
  }

  // =========================================================
  // IMAGE PICK + AI ANALYSIS
  // =========================================================

  Future<void> pickImage(
    ImageSource source,
  ) async {
    try {
      final XFile? image =
          await imagePicker.pickImage(
        source: source,
        imageQuality: 85,
      );

      if (image == null) {
        return;
      }

      setState(() {
        selectedImage = image;

        isAnalyzingImage = true;

        detectedCrop = '';
        detectedDisease = '';

        confidence = 0.0;

        diseaseInfo = null;

        backendAnswer =
            '🌱 ಚಿತ್ರವನ್ನು AI ಪರಿಶೀಲಿಸುತ್ತಿದೆ...';
      });

      final Uint8List imageBytes =
          await image.readAsBytes();

      final request =
          http.MultipartRequest(
        'POST',
        Uri.parse(
          '$backendUrl/upload-image',
        ),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          imageBytes,
          filename: image.name,
        ),
      );

      final response =
          await request.send();

      final responseBody =
          await response.stream
              .bytesToString();

      debugPrint(
        'Image API Status: '
        '${response.statusCode}',
      );

      debugPrint(
        'Image API Response: '
        '$responseBody',
      );

      if (!mounted) {
        return;
      }

      if (response.statusCode == 200) {
        final data =
            jsonDecode(responseBody);

        final String label =
            data['crop_or_disease']
                    ?.toString() ??
                'Unknown';

        final double resultConfidence =
            data['confidence'] is num
                ? (data['confidence'] as num)
                    .toDouble()
                : 0.0;

        final dynamic info =
            data['disease_info'];

        final Map<String, dynamic>?
            resultDiseaseInfo =
            info is Map
                ? Map<String, dynamic>.from(
                    info,
                  )
                : null;

        String crop = '';
        String disease = label;

        if (label.contains('___')) {
          final parts =
              label.split('___');

          crop = parts.first;

          disease =
              parts.sublist(1).join('___');
        }

        setState(() {
          isAnalyzingImage = false;

          detectedCrop =
              formatCropName(crop);

          detectedDisease =
              formatDiseaseName(disease);

          confidence =
              resultConfidence;

          diseaseInfo =
              resultDiseaseInfo;

          backendAnswer =
              '🌱 AI image analysis completed successfully!';
        });
      } else {
        setState(() {
          isAnalyzingImage = false;

          backendAnswer =
              '❌ Image analysis failed.\n'
              'Status: ${response.statusCode}';
        });
      }
    } catch (e) {
      debugPrint(
        'Image Upload Error: $e',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        isAnalyzingImage = false;

        backendAnswer =
            '❌ Image upload error.\n'
            'Backend connection check ಮಾಡಿ.';
      });
    }
  }

  // =========================================================
  // FORMAT CROP
  // =========================================================

  String formatCropName(
    String crop,
  ) {
    if (crop.isEmpty) {
      return 'ಬೆಳೆ ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ';
    }

    switch (crop.toLowerCase()) {
      case 'tomato':
        return 'ಟೊಮ್ಯಾಟೊ (Tomato)';

      case 'pepper,_bell':
        return 'Capsicum / Bell Pepper';

      case 'soybean':
        return 'ಸೋಯಾಬೀನ್ (Soybean)';

      case 'potato':
        return 'ಆಲೂಗಡ್ಡೆ (Potato)';

      case 'corn':
        return 'ಮೆಕ್ಕೆಜೋಳ (Corn)';

      case 'apple':
        return 'ಸೇಬು (Apple)';

      case 'grape':
        return 'ದ್ರಾಕ್ಷಿ (Grape)';

      default:
        return crop.replaceAll(
          '_',
          ' ',
        );
    }
  }

  // =========================================================
  // FORMAT DISEASE
  // =========================================================

  String formatDiseaseName(
    String disease,
  ) {
    String result =
        disease.replaceAll(
      '_',
      ' ',
    );

    result = result.replaceAll(
      'Tomato',
      '',
    );

    result = result.replaceAll(
      'Pepper',
      '',
    );

    result = result.replaceAll(
      'Potato',
      '',
    );

    result = result.trim();

    if (result.toLowerCase() ==
        'healthy') {
      return 'Healthy / ಆರೋಗ್ಯಕರ 🌿';
    }

    if (result.isEmpty) {
      return 'ರೋಗವನ್ನು ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ';
    }

    return result;
  }

  // =========================================================
  // SEND PROBLEM TO BACKEND
  // =========================================================

  Future<void> sendToBackend(
    String text,
  ) async {
    if (text.trim().isEmpty) {
      return;
    }

    try {
      final response =
          await http.post(
        Uri.parse(
          '$backendUrl/problem',
        ),
        headers: {
          'Content-Type':
              'application/json',
        },
        body: jsonEncode({
          'problem': text,
        }),
      );

      if (!mounted) {
        return;
      }

      if (response.statusCode == 200) {
        final data =
            jsonDecode(
          response.body,
        );

        final String problem =
            data['problem']
                    ?.toString() ??
                text;

        final String answer =
            data['answer']
                    ?.toString() ??
                'ಉತ್ತರ ಲಭ್ಯವಿಲ್ಲ.';

        final List<String> advice =
            data['main_advice'] is List
                ? List<String>.from(
                    data['main_advice'],
                  )
                : [];

        final List<String> doList =
            data['do'] is List
                ? List<String>.from(
                    data['do'],
                  )
                : [];

        final List<String> dontList =
            data['dont'] is List
                ? List<String>.from(
                    data['dont'],
                  )
                : [];

        final String next =
            data['next_step']
                    ?.toString() ??
                '';

        setState(() {
          recognizedText = problem;

          backendAnswer = answer;

          mainAdvice = advice;

          doSteps = doList;

          dontSteps = dontList;

          nextStep = next;
        });

        await speakAnswer(answer);
      } else {
        setState(() {
          backendAnswer =
              '❌ Backend response error: '
              '${response.statusCode}';
        });
      }
    } catch (e) {
      debugPrint(
        'Backend Error: $e',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        backendAnswer =
            '❌ Backend connect ಆಗುತ್ತಿಲ್ಲ.\n'
            'FastAPI server running ಇದೆಯೇ check ಮಾಡಿ.';
      });
    }
  }

  // =========================================================
  // START SPEECH
  // =========================================================

  Future<void> startListening() async {
    try {
      final bool available =
          await speech.initialize();

      if (!available) {
        if (!mounted) {
          return;
        }

        setState(() {
          backendAnswer =
              '❌ Microphone / Speech recognition available ಇಲ್ಲ.';
        });

        return;
      }

      setState(() {
        isListening = true;

        recognizedText = '';

        backendAnswer = '';

        mainAdvice = [];

        doSteps = [];

        dontSteps = [];

        nextStep = '';
      });

      await speech.listen(
        listenOptions: stt.SpeechListenOptions(
          localeId: 'kn_IN',
        ),
        onResult: (result) {
          if (!mounted) {
            return;
          }

          if (result.recognizedWords
              .isNotEmpty) {
            setState(() {
              recognizedText =
                  result.recognizedWords;
            });
          }
        },
      );
    } catch (e) {
      debugPrint(
        'Speech Error: $e',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        isListening = false;

        backendAnswer =
            '❌ Voice recognition error.';
      });
    }
  }

  // =========================================================
  // STOP SPEECH
  // =========================================================

  Future<void> stopListening() async {
    await speech.stop();

    if (!mounted) {
      return;
    }

    setState(() {
      isListening = false;
    });

    if (recognizedText
        .trim()
        .isNotEmpty) {
      await sendToBackend(
        recognizedText,
      );
    }
  }

  // =========================================================
  // SECTION TITLE
  // =========================================================

  Widget sectionTitle(
    String title,
    IconData icon,
  ) {
    return Row(
      children: [
        Container(
          padding:
              const EdgeInsets.all(9),
          decoration: BoxDecoration(
            color:
                Colors.green.shade100,
            borderRadius:
                BorderRadius.circular(12),
          ),
          child: Icon(
            icon,
            color:
                Colors.green.shade800,
            size: 22,
          ),
        ),

        const SizedBox(
          width: 10,
        ),

        Expanded(
          child: Text(
            title,
            style:
                const TextStyle(
              fontSize: 18,
              fontWeight:
                  FontWeight.bold,
            ),
          ),
        ),
      ],
    );
  }

  // =========================================================
  // FARMER RESPONSE
  // =========================================================

  Widget buildFarmerResponse() {
    if (recognizedText.isEmpty ||
        backendAnswer.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      children: [
        Container(
          width: double.infinity,
          margin:
              const EdgeInsets.only(
            top: 20,
          ),
          padding:
              const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient:
                LinearGradient(
              colors: [
                Colors.green.shade50,
                Colors.white,
              ],
              begin:
                  Alignment.topLeft,
              end:
                  Alignment.bottomRight,
            ),
            borderRadius:
                BorderRadius.circular(22),
            border: Border.all(
              color:
                  Colors.green.shade200,
            ),
            boxShadow: [
              BoxShadow(
                color:
                    Colors.green.withValues(
                  alpha: 0.08,
                ),
                blurRadius: 12,
                offset:
                    const Offset(0, 5),
              ),
            ],
          ),
          child: Column(
            children: [
              sectionTitle(
                'KrushiVaani ಉತ್ತರ',
                Icons.support_agent,
              ),

              const SizedBox(
                height: 16,
              ),

              Text(
                backendAnswer,
                style:
                    const TextStyle(
                  fontSize: 17,
                  height: 1.55,
                ),
                textAlign:
                    TextAlign.center,
              ),

              const SizedBox(
                height: 14,
              ),

              Container(
                padding:
                    const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 8,
                ),
                decoration:
                    BoxDecoration(
                  color:
                      Colors.green.shade100,
                  borderRadius:
                      BorderRadius.circular(
                    30,
                  ),
                ),
                child: const Row(
                  mainAxisSize:
                      MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.volume_up,
                      size: 18,
                      color:
                          Colors.green,
                    ),
                    SizedBox(
                      width: 6,
                    ),
                    Text(
                      'Kannada Voice Answer',
                      style: TextStyle(
                        fontWeight:
                            FontWeight.w600,
                        color:
                            Colors.green,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),

        if (mainAdvice.isNotEmpty)
          buildAdviceCard(),

        if (doSteps.isNotEmpty)
          buildDoCard(),

        if (dontSteps.isNotEmpty)
          buildDontCard(),

        if (nextStep.isNotEmpty)
          buildNextStepCard(),
      ],
    );
  }

  // =========================================================
  // ADVICE CARD
  // =========================================================

  Widget buildAdviceCard() {
    return buildListCard(
      title:
          'ರೈತರಿಗೆ ಮುಖ್ಯ ಸಲಹೆಗಳು',
      icon:
          Icons.lightbulb_outline,
      color:
          Colors.orange,
      background:
          Colors.orange.shade50,
      items:
          mainAdvice,
    );
  }

  // =========================================================
  // DO CARD
  // =========================================================

  Widget buildDoCard() {
    return buildListCard(
      title:
          'ರೈತರು ಮಾಡಬೇಕಾದದ್ದು',
      icon:
          Icons.check_circle_outline,
      color:
          Colors.green,
      background:
          Colors.green.shade50,
      items:
          doSteps,
    );
  }

  // =========================================================
  // DON'T CARD
  // =========================================================

  Widget buildDontCard() {
    return buildListCard(
      title:
          'ರೈತರು ಮಾಡಬಾರದದ್ದು',
      icon:
          Icons.block,
      color:
          Colors.red,
      background:
          Colors.red.shade50,
      items:
          dontSteps,
    );
  }

  // =========================================================
  // COMMON LIST CARD
  // =========================================================

  Widget buildListCard({
    required String title,
    required IconData icon,
    required Color color,
    required Color background,
    required List<String> items,
  }) {
    return Container(
      width: double.infinity,
      margin:
          const EdgeInsets.only(
        top: 14,
      ),
      padding:
          const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: background,
        borderRadius:
            BorderRadius.circular(20),
        border: Border.all(
          color:
              color.withValues(
            alpha: 0.25,
          ),
        ),
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                icon,
                color: color,
              ),

              const SizedBox(
                width: 9,
              ),

              Expanded(
                child: Text(
                  title,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight:
                        FontWeight.bold,
                    color: color,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(
            height: 14,
          ),

          ...items
              .asMap()
              .entries
              .map(
            (entry) {
              return Container(
                margin:
                    const EdgeInsets.only(
                  bottom: 9,
                ),
                padding:
                    const EdgeInsets.all(
                  11,
                ),
                decoration:
                    BoxDecoration(
                  color: Colors.white
                      .withValues(
                    alpha: 0.65,
                  ),
                  borderRadius:
                      BorderRadius.circular(
                    12,
                  ),
                ),
                child: Row(
                  crossAxisAlignment:
                      CrossAxisAlignment
                          .start,
                  children: [
                    CircleAvatar(
                      radius: 12,
                      backgroundColor:
                          color,
                      child: Text(
                        '${entry.key + 1}',
                        style:
                            const TextStyle(
                          color:
                              Colors.white,
                          fontSize: 12,
                          fontWeight:
                              FontWeight.bold,
                        ),
                      ),
                    ),

                    const SizedBox(
                      width: 10,
                    ),

                    Expanded(
                      child: Text(
                        entry.value,
                        style:
                            const TextStyle(
                          fontSize: 15.5,
                          height: 1.45,
                        ),
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  // =========================================================
  // NEXT STEP
  // =========================================================

  Widget buildNextStepCard() {
    return Container(
      width: double.infinity,
      margin:
          const EdgeInsets.only(
        top: 16,
      ),
      padding:
          const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius:
            BorderRadius.circular(20),
        border: Border.all(
          color:
              Colors.blue.shade200,
        ),
      ),
      child: Column(
        children: [
          sectionTitle(
            'ಮುಂದಿನ ಹಂತ',
            Icons.camera_alt_outlined,
          ),

          const SizedBox(
            height: 14,
          ),

          Text(
            nextStep,
            style:
                const TextStyle(
              fontSize: 15.5,
              height: 1.5,
            ),
            textAlign:
                TextAlign.center,
          ),

          const SizedBox(
            height: 16,
          ),

          Row(
            children: [
              Expanded(
                child:
                    OutlinedButton.icon(
                  onPressed:
                      isAnalyzingImage
                          ? null
                          : () {
                              pickImage(
                                ImageSource
                                    .camera,
                              );
                            },
                  icon:
                      const Icon(
                    Icons.camera_alt,
                  ),
                  label:
                      const Text(
                    'Camera',
                  ),
                ),
              ),

              const SizedBox(
                width: 12,
              ),

              Expanded(
                child:
                    ElevatedButton.icon(
                  onPressed:
                      isAnalyzingImage
                          ? null
                          : () {
                              pickImage(
                                ImageSource
                                    .gallery,
                              );
                            },
                  icon:
                      const Icon(
                    Icons.photo_library,
                  ),
                  label:
                      const Text(
                    'Gallery',
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // =========================================================
  // FARMER ADVICE AFTER IMAGE
  // =========================================================

  String getFarmerAdvice() {
    if (confidence < 50) {
      return '⚠️ AI ಫಲಿತಾಂಶದ confidence ಕಡಿಮೆ ಇದೆ.\n\n'
          'ಈ ಫಲಿತಾಂಶವನ್ನು ಖಚಿತ ರೋಗನಿರ್ಣಯ ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ. '
          'ಸ್ಪಷ್ಟವಾದ ಚಿತ್ರವನ್ನು ತೆಗೆದು ಮತ್ತೊಮ್ಮೆ ಪರಿಶೀಲಿಸಿ. '
          'ರೋಗ ಖಚಿತವಾಗುವ ಮೊದಲು ಯಾವುದೇ ಔಷಧಿಯನ್ನು ಬಳಸಬೇಡಿ.';
    }

    if (detectedDisease
        .toLowerCase()
        .contains('healthy')) {
      return '🌿 ನಿಮ್ಮ ಬೆಳೆಯ ಚಿತ್ರ ಆರೋಗ್ಯಕರವಾಗಿರುವ '
          'ಸಾಧ್ಯತೆ ಇದೆ.\n\n'
          'ಆದರೂ ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.';
    }

    return '🌱 ಬೆಳೆಯಲ್ಲಿ ಸಮಸ್ಯೆ ಇರುವ ಸಾಧ್ಯತೆ ಇದೆ.\n\n'
        'ಸ್ಪಷ್ಟವಾದ ಚಿತ್ರದಿಂದ ಮತ್ತೊಮ್ಮೆ ಪರಿಶೀಲಿಸಿ '
        'ಮತ್ತು ಅಗತ್ಯವಿದ್ದರೆ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ.';
  }

  // =========================================================
  // AI RESULT
  // =========================================================

  Widget buildDiseaseResultCard() {
    if (detectedDisease.isEmpty) {
      return const SizedBox.shrink();
    }

    final bool lowConfidence =
        confidence < 50;

    final bool isHealthy =
        detectedDisease
            .toLowerCase()
            .contains('healthy');

    final double progress =
        (confidence / 100)
            .clamp(0.0, 1.0);

    return Container(
      width: double.infinity,
      margin:
          const EdgeInsets.only(
        top: 20,
      ),
      padding:
          const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius:
            BorderRadius.circular(24),
        border: Border.all(
          color:
              Colors.green.shade200,
        ),
        boxShadow: [
          BoxShadow(
            color:
                Colors.green.withValues(
              alpha: 0.08,
            ),
            blurRadius: 15,
            offset:
                const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        children: [
          sectionTitle(
            'KrushiVaani AI Result',
            Icons.auto_awesome,
          ),

          const SizedBox(
            height: 20,
          ),

          Container(
            padding:
                const EdgeInsets.all(16),
            decoration:
                BoxDecoration(
              color:
                  Colors.green.shade50,
              shape:
                  BoxShape.circle,
            ),
            child: Icon(
              isHealthy
                  ? Icons.eco
                  : Icons.local_florist,
              size: 50,
              color:
                  Colors.green,
            ),
          ),

          const SizedBox(
            height: 16,
          ),

          Text(
            detectedCrop.isEmpty
                ? 'Unknown'
                : detectedCrop,
            style:
                const TextStyle(
              fontSize: 21,
              fontWeight:
                  FontWeight.bold,
            ),
            textAlign:
                TextAlign.center,
          ),

          const SizedBox(
            height: 18,
          ),

          Container(
            width: double.infinity,
            padding:
                const EdgeInsets.all(15),
            decoration:
                BoxDecoration(
              color: isHealthy
                  ? Colors.green.shade50
                  : Colors.red.shade50,
              borderRadius:
                  BorderRadius.circular(
                16,
              ),
            ),
            child: Column(
              children: [
                Text(
                  'ಗುರುತಿಸಲಾದ ಸಮಸ್ಯೆ',
                  style: TextStyle(
                    color:
                        Colors.grey.shade700,
                  ),
                ),

                const SizedBox(
                  height: 6,
                ),

                Text(
                  detectedDisease,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight:
                        FontWeight.bold,
                    color: isHealthy
                        ? Colors.green.shade700
                        : Colors.red.shade700,
                  ),
                  textAlign:
                      TextAlign.center,
                ),
              ],
            ),
          ),

          const SizedBox(
            height: 20,
          ),

          Row(
            mainAxisAlignment:
                MainAxisAlignment
                    .spaceBetween,
            children: [
              const Text(
                'AI Confidence',
                style: TextStyle(
                  fontWeight:
                      FontWeight.w600,
                ),
              ),

              Text(
                '${confidence.toStringAsFixed(2)}%',
                style:
                    const TextStyle(
                  fontSize: 20,
                  fontWeight:
                      FontWeight.bold,
                ),
              ),
            ],
          ),

          const SizedBox(
            height: 9,
          ),

          ClipRRect(
            borderRadius:
                BorderRadius.circular(
              20,
            ),
            child:
                LinearProgressIndicator(
              value: progress,
              minHeight: 10,
              backgroundColor:
                  Colors.grey.shade200,
              color: lowConfidence
                  ? Colors.orange
                  : Colors.green,
            ),
          ),

          const SizedBox(
            height: 12,
          ),

          Container(
            width: double.infinity,
            padding:
                const EdgeInsets.all(13),
            decoration:
                BoxDecoration(
              color: lowConfidence
                  ? Colors.orange.shade50
                  : Colors.green.shade50,
              borderRadius:
                  BorderRadius.circular(
                14,
              ),
            ),
            child: Text(
              lowConfidence
                  ? '⚠️ Low Confidence — ಮತ್ತೊಂದು ಸ್ಪಷ್ಟವಾದ ಚಿತ್ರದಿಂದ ಪರಿಶೀಲಿಸಿ.'
                  : '✅ Good Confidence — ಆದರೂ ಇದು ಅಂತಿಮ diagnosis ಅಲ್ಲ.',
              style: TextStyle(
                fontSize: 14.5,
                fontWeight:
                    FontWeight.w600,
                color: lowConfidence
                    ? Colors.orange.shade800
                    : Colors.green.shade800,
              ),
              textAlign:
                  TextAlign.center,
            ),
          ),

          const SizedBox(
            height: 16,
          ),

          Container(
            width: double.infinity,
            padding:
                const EdgeInsets.all(15),
            decoration:
                BoxDecoration(
              color:
                  Colors.blue.shade50,
              borderRadius:
                  BorderRadius.circular(
                15,
              ),
              border: Border.all(
                color:
                    Colors.blue.shade100,
              ),
            ),
            child: Text(
              getFarmerAdvice(),
              style:
                  const TextStyle(
                fontSize: 15,
                height: 1.5,
              ),
              textAlign:
                  TextAlign.center,
            ),
          ),

          buildDiseaseInfoSection(),

          const SizedBox(
            height: 18,
          ),

          const Text(
            '⚠️ AI ಫಲಿತಾಂಶವನ್ನು ಖಚಿತ ರೋಗನಿರ್ಣಯ ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ. ಅಗತ್ಯವಿದ್ದರೆ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ.',
            style: TextStyle(
              fontSize: 13.5,
              fontWeight:
                  FontWeight.w600,
              height: 1.4,
            ),
            textAlign:
                TextAlign.center,
          ),
        ],
      ),
    );
  }

  // =========================================================
  // DISEASE INFORMATION
  // =========================================================

  Widget buildDiseaseInfoSection() {
    if (diseaseInfo == null) {
      return const SizedBox.shrink();
    }

    final String summary =
        diseaseInfo!['summary']
                ?.toString() ??
            '';

    final List<dynamic> doList =
        diseaseInfo!['do'] is List
            ? List<dynamic>.from(
                diseaseInfo!['do'],
              )
            : [];

    final List<dynamic> dontList =
        diseaseInfo!['dont'] is List
            ? List<dynamic>.from(
                diseaseInfo!['dont'],
              )
            : [];

    return Column(
      children: [
        const SizedBox(
          height: 18,
        ),

        if (summary.isNotEmpty)
          Container(
            width: double.infinity,
            padding:
                const EdgeInsets.all(16),
            decoration:
                BoxDecoration(
              color:
                  Colors.blue.shade50,
              borderRadius:
                  BorderRadius.circular(
                16,
              ),
              border: Border.all(
                color:
                    Colors.blue.shade100,
              ),
            ),
            child: Column(
              children: [
                sectionTitle(
                  'ರೋಗದ ಬಗ್ಗೆ',
                  Icons.info_outline,
                ),

                const SizedBox(
                  height: 12,
                ),

                Text(
                  summary,
                  style:
                      const TextStyle(
                    fontSize: 15.5,
                    height: 1.5,
                  ),
                  textAlign:
                      TextAlign.center,
                ),
              ],
            ),
          ),

        if (doList.isNotEmpty) ...[
          const SizedBox(
            height: 14,
          ),

          buildInfoListCard(
            title:
                'ರೈತರು ಮಾಡಬೇಕಾದದ್ದು',
            items:
                doList,
            color:
                Colors.green,
            background:
                Colors.green.shade50,
          ),
        ],

        if (dontList.isNotEmpty) ...[
          const SizedBox(
            height: 14,
          ),

          buildInfoListCard(
            title:
                'ರೈತರು ಮಾಡಬಾರದದ್ದು',
            items:
                dontList,
            color:
                Colors.red,
            background:
                Colors.red.shade50,
          ),
        ],
      ],
    );
  }

  // =========================================================
  // INFO LIST CARD
  // =========================================================

  Widget buildInfoListCard({
    required String title,
    required List<dynamic> items,
    required Color color,
    required Color background,
  }) {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: background,
        borderRadius:
            BorderRadius.circular(16),
        border: Border.all(
          color:
              color.withValues(
            alpha: 0.25,
          ),
        ),
      ),
      child: Column(
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: 18,
              fontWeight:
                  FontWeight.bold,
              color: color,
            ),
            textAlign:
                TextAlign.center,
          ),

          const SizedBox(
            height: 12,
          ),

          ...items
              .asMap()
              .entries
              .map(
            (entry) {
              return Container(
                width:
                    double.infinity,
                margin:
                    const EdgeInsets.only(
                  bottom: 8,
                ),
                padding:
                    const EdgeInsets.all(
                  10,
                ),
                decoration:
                    BoxDecoration(
                  color: Colors.white
                      .withValues(
                    alpha: 0.65,
                  ),
                  borderRadius:
                      BorderRadius.circular(
                    11,
                  ),
                ),
                child: Text(
                  '${entry.key + 1}. ${entry.value}',
                  style:
                      const TextStyle(
                    fontSize: 15,
                    height: 1.4,
                  ),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  // =========================================================
  // IMAGE PREVIEW
  // =========================================================

  Widget buildImagePreview() {
    if (selectedImage == null) {
      return const SizedBox.shrink();
    }

    return FutureBuilder<Uint8List>(
      future:
          selectedImage!.readAsBytes(),
      builder:
          (context, snapshot) {
        if (snapshot.connectionState ==
                ConnectionState.done &&
            snapshot.hasData) {
          return Container(
            margin:
                const EdgeInsets.only(
              top: 20,
            ),
            padding:
                const EdgeInsets.all(12),
            decoration:
                BoxDecoration(
              color:
                  Colors.white,
              borderRadius:
                  BorderRadius.circular(
                20,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black
                      .withValues(
                    alpha: 0.06,
                  ),
                  blurRadius: 12,
                  offset:
                      const Offset(0, 4),
                ),
              ],
            ),
            child: Column(
              children: [
                ClipRRect(
                  borderRadius:
                      BorderRadius.circular(
                    15,
                  ),
                  child: Image.memory(
                    snapshot.data!,
                    height: 230,
                    width:
                        double.infinity,
                    fit: BoxFit.cover,
                  ),
                ),

                const SizedBox(
                  height: 10,
                ),

                const Row(
                  mainAxisAlignment:
                      MainAxisAlignment
                          .center,
                  children: [
                    Icon(
                      Icons.check_circle,
                      color:
                          Colors.green,
                      size: 20,
                    ),

                    SizedBox(
                      width: 7,
                    ),

                    Text(
                      'ಬೆಳೆ ಚಿತ್ರ ಆಯ್ಕೆ ಮಾಡಲಾಗಿದೆ',
                      style: TextStyle(
                        fontSize: 15.5,
                        fontWeight:
                            FontWeight.bold,
                        color:
                            Colors.green,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          );
        }

        return const Padding(
          padding:
              EdgeInsets.all(20),
          child:
              CircularProgressIndicator(),
        );
      },
    );
  }

  // =========================================================
  // AI LOADING CARD
  // =========================================================

  Widget buildLoadingCard() {
    if (!isAnalyzingImage) {
      return const SizedBox.shrink();
    }

    return Container(
      width: double.infinity,
      margin:
          const EdgeInsets.only(
        top: 18,
      ),
      padding:
          const EdgeInsets.all(22),
      decoration:
          BoxDecoration(
        color: Colors.white,
        borderRadius:
            BorderRadius.circular(20),
        border: Border.all(
          color:
              Colors.green.shade200,
        ),
      ),
      child: const Column(
        children: [
          SizedBox(
            height: 45,
            width: 45,
            child:
                CircularProgressIndicator(
              strokeWidth: 4,
            ),
          ),

          SizedBox(
            height: 16,
          ),

          Text(
            '🤖 AI ಬೆಳೆಯ ಚಿತ್ರವನ್ನು ಪರಿಶೀಲಿಸುತ್ತಿದೆ...',
            style:
                TextStyle(
              fontSize: 17,
              fontWeight:
                  FontWeight.bold,
            ),
            textAlign:
                TextAlign.center,
          ),

          SizedBox(
            height: 7,
          ),

          Text(
            'ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯ ಕಾಯಿರಿ',
            style:
                TextStyle(
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }

  // =========================================================
  // VOICE BUTTON
  // =========================================================

  Widget buildVoiceButton() {
    return GestureDetector(
      onTap: isListening
          ? stopListening
          : startListening,
      child: AnimatedContainer(
        duration:
            const Duration(
          milliseconds: 250,
        ),
        width: 150,
        height: 150,
        decoration:
            BoxDecoration(
          shape:
              BoxShape.circle,
          color: isListening
              ? Colors.red.shade400
              : Colors.green.shade600,
          boxShadow: [
            BoxShadow(
              color: (isListening
                      ? Colors.red
                      : Colors.green)
                  .withValues(
                alpha: 0.25,
              ),
              blurRadius: 25,
              spreadRadius:
                  isListening ? 8 : 2,
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment:
              MainAxisAlignment.center,
          children: [
            Icon(
              isListening
                  ? Icons.stop
                  : Icons.mic,
              color:
                  Colors.white,
              size: 52,
            ),

            const SizedBox(
              height: 8,
            ),

            Text(
              isListening
                  ? 'STOP'
                  : 'SPEAK',
              style:
                  const TextStyle(
                color:
                    Colors.white,
                fontWeight:
                    FontWeight.bold,
                fontSize: 15,
                letterSpacing: 1,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // =========================================================
  // HOME BUILD
  // =========================================================

  @override
  Widget build(
    BuildContext context,
  ) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        backgroundColor:
            Colors.green.shade700,
        foregroundColor:
            Colors.white,
        title: const Row(
          mainAxisSize:
              MainAxisSize.min,
          children: [
            Icon(
              Icons.agriculture,
              size: 27,
            ),

            SizedBox(
              width: 8,
            ),

            Text(
              'KrushiVaani',
              style:
                  TextStyle(
                fontWeight:
                    FontWeight.bold,
                fontSize: 21,
              ),
            ),
          ],
        ),
        centerTitle: true,
      ),

      body: SafeArea(
        child:
            SingleChildScrollView(
          padding:
              const EdgeInsets.fromLTRB(
            18,
            22,
            18,
            35,
          ),
          child: Column(
            children: [
              // =================================================
              // HERO
              // =================================================

              Container(
                width:
                    double.infinity,
                padding:
                    const EdgeInsets.all(
                  22,
                ),
                decoration:
                    BoxDecoration(
                  gradient:
                      LinearGradient(
                    colors: [
                      Colors.green.shade700,
                      Colors.green.shade500,
                    ],
                    begin:
                        Alignment.topLeft,
                    end:
                        Alignment.bottomRight,
                  ),
                  borderRadius:
                      BorderRadius.circular(
                    28,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.green
                          .withValues(
                        alpha: 0.18,
                      ),
                      blurRadius: 18,
                      offset:
                          const Offset(0, 7),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    Container(
                      padding:
                          const EdgeInsets
                              .all(15),
                      decoration:
                          BoxDecoration(
                        color:
                            Colors.white
                                .withValues(
                          alpha: 0.18,
                        ),
                        shape:
                            BoxShape.circle,
                      ),
                      child:
                          const Icon(
                        Icons.agriculture,
                        size: 62,
                        color:
                            Colors.white,
                      ),
                    ),

                    const SizedBox(
                      height: 15,
                    ),

                    const Text(
                      'ನಮಸ್ಕಾರ ರೈತ ಸ್ನೇಹಿತರೆ! 🌾',
                      style:
                          TextStyle(
                        color:
                            Colors.white,
                        fontSize: 23,
                        fontWeight:
                            FontWeight.bold,
                      ),
                      textAlign:
                          TextAlign.center,
                    ),

                    const SizedBox(
                      height: 8,
                    ),

                    Text(
                      'ನಿಮ್ಮ ಕೃಷಿ ಸಮಸ್ಯೆಗೆ AI ಸಹಾಯ',
                      style:
                          TextStyle(
                        color:
                            Colors.white
                                .withValues(
                          alpha: 0.92,
                        ),
                        fontSize: 16,
                      ),
                      textAlign:
                          TextAlign.center,
                    ),
                  ],
                ),
              ),

              const SizedBox(
                height: 24,
              ),

              // =================================================
// FARMER PROFILE CARD
// =================================================

SizedBox(
  width: double.infinity,
  child: Card(
    elevation: 3,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(18),
    ),
    child: InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) =>
                const FarmerProfilePage(),
          ),
        );
      },
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.green.shade50,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.person,
                color: Colors.green,
                size: 30,
              ),
            ),

            const SizedBox(width: 14),

            const Expanded(
              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start,
                children: [
                  Text(
                    'ರೈತ ಪ್ರೊಫೈಲ್ 👨‍🌾',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  SizedBox(height: 5),

                  Text(
                    'ನಿಮ್ಮ ಹೆಸರು, ಗ್ರಾಮ ಮತ್ತು ಬೆಳೆ ಮಾಹಿತಿಯನ್ನು ಉಳಿಸಿ',
                    style: TextStyle(
                      fontSize: 13.5,
                    ),
                  ),
                ],
              ),
            ),

            const Icon(
              Icons.arrow_forward_ios,
              size: 18,
              color: Colors.green,
            ),
          ],
        ),
      ),
    ),
  ),
),

const SizedBox(height: 18),

              // =================================================
              // GOVERNMENT SCHEMES CARD
              // =================================================

              SizedBox(
                width:
                    double.infinity,
                child: Card(


                  elevation: 3,
                  shape:
                      RoundedRectangleBorder(
                    borderRadius:
                        BorderRadius.circular(
                      18,
                    ),
                  ),
                  child:
                      InkWell(
                    borderRadius:
                        BorderRadius.circular(
                      18,
                    ),
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder:
                              (context) =>
                                  const GovernmentSchemesPage(),
                        ),
                      );
                    },
                    child: Padding(
                      padding:
                          const EdgeInsets
                              .all(18),
                      child: Row(
                        children: [
                          Container(
                            padding:
                                const EdgeInsets
                                    .all(12),
                            decoration:
                                BoxDecoration(
                              color: Colors
                                  .green
                                  .shade50,
                              shape:
                                  BoxShape
                                      .circle,
                            ),
                            child:
                                const Icon(
                              Icons
                                  .account_balance,
                              color:
                                  Colors.green,
                              size: 30,
                            ),
                          ),

                          const SizedBox(
                            width: 14,
                          ),

                          const Expanded(
                            child: Column(
                              crossAxisAlignment:
                                  CrossAxisAlignment
                                      .start,
                              children: [
                                Text(
                                  'ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು 🌾',
                                  style:
                                      TextStyle(
                                    fontSize:
                                        18,
                                    fontWeight:
                                        FontWeight
                                            .bold,
                                  ),
                                ),

                                SizedBox(
                                  height: 5,
                                ),

                                Text(
                                  'ರೈತರಿಗೆ ಲಭ್ಯವಿರುವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಮತ್ತು ಸೌಲಭ್ಯಗಳನ್ನು ನೋಡಿ',
                                  style:
                                      TextStyle(
                                    fontSize:
                                        13.5,
                                  ),
                                ),
                              ],
                            ),
                          ),

                          const Icon(
                            Icons
                                .arrow_forward_ios,
                            size: 18,
                            color:
                                Colors.green,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),

              const SizedBox(
                height: 18,
              ),

              // =================================================
// FARMER NOTIFICATIONS CARD
// =================================================

SizedBox(
  width: double.infinity,
  child: Card(
    elevation: 3,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(18),
    ),
    child: InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) =>
                const FarmerNotificationsPage(),
          ),
        );
      },
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.orange.shade50,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.notifications,
                color: Colors.orange,
                size: 30,
              ),
            ),

            const SizedBox(width: 14),

            const Expanded(
              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start,
                children: [
                  Text(
                    'ರೈತ ಅಧಿಸೂಚನೆಗಳು 🔔',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  SizedBox(height: 5),

                  Text(
                    'ಹೊಸ ರೈತ ಯೋಜನೆಗಳು ಮತ್ತು ಕೃಷಿ ಮಾಹಿತಿಯ updates ನೋಡಿ',
                    style: TextStyle(
                      fontSize: 13.5,
                    ),
                  ),
                ],
              ),
            ),

            const Icon(
              Icons.arrow_forward_ios,
              size: 18,
              color: Colors.orange,
            ),
          ],
        ),
      ),
    ),
  ),
),

const SizedBox(
  height: 18,
),

const SizedBox(height: 12),

SizedBox(
  width: double.infinity,
  child: Card(
    elevation: 3,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(18),
    ),
    child: InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => const WeatherPage(),
          ),
        );
      },
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.cloud,
                color: Colors.blue,
                size: 30,
              ),
            ),
            const SizedBox(width: 14),
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'ಹವಾಮಾನ ಮಾಹಿತಿ 🌦️',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 5),
                  Text(
                    'ಇಂದಿನ ಹವಾಮಾನ ಮತ್ತು ರೈತರಿಗೆ ಉಪಯುಕ್ತ ಮಾಹಿತಿ ನೋಡಿ',
                    style: TextStyle(
                      fontSize: 13.5,
                    ),
                  ),
                ],
              ),
            ),
            const Icon(
              Icons.arrow_forward_ios,
              size: 18,
              color: Colors.blue,
            ),
          ],
        ),
      ),
    ),
  ),
),
              // =================================================
              // AGRICULTURE MEDICINES CARD
              // =================================================

              SizedBox(
                width:
                    double.infinity,
                child: Card(
                  elevation: 3,
                  shape:
                      RoundedRectangleBorder(
                    borderRadius:
                        BorderRadius.circular(
                      18,
                    ),
                  ),
                  child:
                      InkWell(
                    borderRadius:
                        BorderRadius.circular(
                      18,
                    ),
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder:
                              (context) =>
                                  const AgricultureMedicinePage(),
                        ),
                      );
                    },
                    child: Padding(
                      padding:
                          const EdgeInsets
                              .all(18),
                      child: Row(
                        children: [
                          Container(
                            padding:
                                const EdgeInsets
                                    .all(12),
                            decoration:
                                BoxDecoration(
                              color: Colors
                                  .green
                                  .shade50,
                              shape:
                                  BoxShape
                                      .circle,
                            ),
                            child:
                                const Icon(
                              Icons.medication,
                              color:
                                  Colors.green,
                              size: 30,
                            ),
                          ),

                          const SizedBox(
                            width: 14,
                          ),

                          const Expanded(
                            child: Column(
                              crossAxisAlignment:
                                  CrossAxisAlignment
                                      .start,
                              children: [
                                Text(
                                  'ಕೃಷಿ ಔಷಧಿ ಮಾಹಿತಿ 💊',
                                  style:
                                      TextStyle(
                                    fontSize:
                                        18,
                                    fontWeight:
                                        FontWeight
                                            .bold,
                                  ),
                                ),

                                SizedBox(
                                  height: 5,
                                ),

                                Text(
                                  'ಬೆಳೆ ರೋಗ ಮತ್ತು ಕೀಟ ಸಮಸ್ಯೆಗಳ ಕುರಿತು ಔಷಧಿ ಮಾಹಿತಿಯನ್ನು ತಿಳಿಯಿರಿ',
                                  style:
                                      TextStyle(
                                    fontSize:
                                        13.5,
                                  ),
                                ),
                              ],
                            ),
                          ),

                          const Icon(
                            Icons
                                .arrow_forward_ios,
                            size: 18,
                            color:
                                Colors.green,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),

              const SizedBox(
                height: 24,
              ),

              // =================================================
// CROP ADVISORY CARD
// =================================================

const SizedBox(height: 18),

SizedBox(
  width: double.infinity,
  child: Card(
    elevation: 3,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(18),
    ),
    child: InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => const CropAdvisoryPage(),
          ),
        );
      },
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.green.shade50,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.agriculture,
                color: Colors.green,
                size: 30,
              ),
            ),

            const SizedBox(width: 14),

            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'ಬೆಳೆ ಸಲಹೆ 🌾',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 5),
                  Text(
                    'ನಿಮ್ಮ ಬೆಳೆಗೆ ಬೇಕಾದ ಕೃಷಿ ಸಲಹೆ ಪಡೆಯಿರಿ',
                    style: TextStyle(
                      fontSize: 13.5,
                    ),
                  ),
                ],
              ),
            ),

            const Icon(
              Icons.arrow_forward_ios,
              size: 18,
              color: Colors.green,
            ),
          ],
        ),
      ),
    ),
  ),
),

              // =================================================
              // VOICE ASSISTANT
              // =================================================

              sectionTitle(
                isListening
                    ? 'ನಿಮ್ಮ ಮಾತನ್ನು ಕೇಳುತ್ತಿದೆ...'
                    : 'ಕನ್ನಡದಲ್ಲಿ ನಿಮ್ಮ ಸಮಸ್ಯೆ ಹೇಳಿ',
                Icons.mic_none,
              ),

              const SizedBox(
                height: 20,
              ),

              buildVoiceButton(),

              const SizedBox(
                height: 15,
              ),

              Text(
                isListening
                    ? 'ಮಾತನಾಡುವುದನ್ನು ಮುಗಿಸಿದ ನಂತರ STOP ಒತ್ತಿರಿ'
                    : 'ಮಾತನಾಡಲು microphone ಒತ್ತಿರಿ',
                style:
                    TextStyle(
                  fontSize: 15,
                  color:
                      Colors.grey.shade700,
                  fontWeight:
                      FontWeight.w500,
                ),
                textAlign:
                    TextAlign.center,
              ),

              // =================================================
              // QUESTION
              // =================================================

              if (recognizedText
                  .isNotEmpty)
                Container(
                  width:
                      double.infinity,
                  margin:
                      const EdgeInsets.only(
                    top: 20,
                  ),
                  padding:
                      const EdgeInsets.all(
                    17,
                  ),
                  decoration:
                      BoxDecoration(
                    color:
                        Colors.white,
                    borderRadius:
                        BorderRadius.circular(
                      18,
                    ),
                    border: Border.all(
                      color:
                          Colors.green.shade200,
                    ),
                  ),
                  child: Column(
                    children: [
                      sectionTitle(
                        'ನಿಮ್ಮ ಪ್ರಶ್ನೆ',
                        Icons
                            .chat_bubble_outline,
                      ),

                      const SizedBox(
                        height: 12,
                      ),

                      Text(
                        recognizedText,
                        style:
                            const TextStyle(
                          fontSize: 17,
                          height: 1.5,
                          fontWeight:
                              FontWeight.w500,
                        ),
                        textAlign:
                            TextAlign.center,
                      ),
                    ],
                  ),
                ),

              // =================================================
              // RESPONSE
              // =================================================

              buildFarmerResponse(),

              // =================================================
              // IMAGE PREVIEW
              // =================================================

              buildImagePreview(),

              // =================================================
              // LOADING
              // =================================================

              buildLoadingCard(),

              // =================================================
              // AI RESULT
              // =================================================

              buildDiseaseResultCard(),

              const SizedBox(
                height: 20,
              ),

              // =================================================
              // IMAGE ANALYSIS
              // =================================================

              Container(
                width:
                    double.infinity,
                padding:
                    const EdgeInsets.all(
                  18,
                ),
                decoration:
                    BoxDecoration(
                  color:
                      Colors.white,
                  borderRadius:
                      BorderRadius.circular(
                    20,
                  ),
                  border: Border.all(
                    color:
                        Colors.green.shade100,
                  ),
                ),
                child: Column(
                  children: [
                    sectionTitle(
                      'ಬೆಳೆ ಚಿತ್ರ ಪರಿಶೀಲನೆ',
                      Icons.image_search,
                    ),

                    const SizedBox(
                      height: 8,
                    ),

                    Text(
                      'ಬೆಳೆಯ ಎಲೆ ಅಥವಾ ಸಮಸ್ಯೆಯ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆ ಮಾಡಿ AI ಮೂಲಕ ಪರಿಶೀಲಿಸಿ.',
                      style:
                          TextStyle(
                        fontSize: 14.5,
                        color: Colors
                            .grey.shade700,
                        height: 1.4,
                      ),
                      textAlign:
                          TextAlign.center,
                    ),

                    const SizedBox(
                      height: 15,
                    ),

                    Row(
                      children: [
                        Expanded(
                          child:
                              OutlinedButton
                                  .icon(
                            onPressed:
                                isAnalyzingImage
                                    ? null
                                    : () {
                                        pickImage(
                                          ImageSource
                                              .camera,
                                        );
                                      },
                            icon:
                                const Icon(
                              Icons
                                  .camera_alt,
                            ),
                            label:
                                const Text(
                              'Camera',
                            ),
                          ),
                        ),

                        const SizedBox(
                          width: 10,
                        ),

                        Expanded(
                          child:
                              ElevatedButton
                                  .icon(
                            onPressed:
                                isAnalyzingImage
                                    ? null
                                    : () {
                                        pickImage(
                                          ImageSource
                                              .gallery,
                                        );
                                      },
                            icon:
                                const Icon(
                              Icons
                                  .photo_library,
                            ),
                            label:
                                const Text(
                              'Gallery',
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(
                height: 22,
              ),

              // =================================================
              // FOOTER
              // =================================================

              Text(
                '🌾 KrushiVaani • AI-powered Farmer Assistant',
                style:
                    TextStyle(
                  fontSize: 13,
                  color:
                      Colors.grey.shade600,
                  fontWeight:
                      FontWeight.w500,
                ),
                textAlign:
                    TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// =========================================================
// GOVERNMENT SCHEMES PAGE
// =========================================================

class GovernmentSchemesPage extends StatefulWidget {
  const GovernmentSchemesPage({super.key});

  @override
  State<GovernmentSchemesPage> createState() =>
      _GovernmentSchemesPageState();
}

class _GovernmentSchemesPageState
    extends State<GovernmentSchemesPage> {
  static const String backendUrl =
      'http://192.168.1.39:8000';

  bool isLoading = true;

  List<dynamic> schemes = [];

  @override
  void initState() {
    super.initState();
    fetchSchemes();
  }

  // =========================================================
  // FETCH GOVERNMENT SCHEMES
  // =========================================================

  Future<void> fetchSchemes() async {
    try {
      final response = await http.get(
        Uri.parse('$backendUrl/schemes'),
      );

      if (!mounted) {
        return;
      }

      if (response.statusCode == 200) {
        final data = jsonDecode(
          response.body,
        );

        setState(() {
          schemes = data['schemes'] ?? [];
          isLoading = false;
        });
      } else {
        setState(() {
          isLoading = false;
        });
      }
    } catch (e) {
      debugPrint(
        'Schemes Error: $e',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        isLoading = false;
      });
    }
  }

  // =========================================================
  // OPEN OFFICIAL WEBSITE
  // =========================================================

  Future<void> openOfficialWebsite(
    String? url,
  ) async {
    if (url == null ||
        url.trim().isEmpty) {
      return;
    }

    try {
      final Uri uri =
          Uri.parse(url.trim());

      final bool launched =
          await launchUrl(
        uri,
        mode:
            LaunchMode.externalApplication,
      );

      if (!launched && mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(
          const SnackBar(
            content: Text(
              'Website open ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.',
            ),
          ),
        );
      }
    } catch (e) {
      debugPrint(
        'Website Error: $e',
      );

      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(
          const SnackBar(
            content: Text(
              'Website open ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.',
            ),
          ),
        );
      }
    }
  }

  // =========================================================
  // BUILD
  // =========================================================

  @override
  Widget build(
    BuildContext context,
  ) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Government Schemes',
        ),
        backgroundColor:
            Colors.green,
        foregroundColor:
            Colors.white,
      ),

      body: isLoading
          ? const Center(
              child:
                  CircularProgressIndicator(),
            )
          : schemes.isEmpty
              ? const Center(
                  child: Text(
                    'ಯಾವುದೇ ಯೋಜನೆಗಳು ಲಭ್ಯವಿಲ್ಲ.',
                  ),
                )
              : RefreshIndicator(
                  onRefresh:
                      fetchSchemes,

                  child:
                      ListView.builder(
                    padding:
                        const EdgeInsets.all(
                      16,
                    ),

                    itemCount:
                        schemes.length,

                    itemBuilder:
                        (context, index) {
                      final scheme =
                          schemes[index];

                      return Card(
                        margin:
                            const EdgeInsets
                                .only(
                          bottom: 16,
                        ),

                        elevation: 3,

                        shape:
                            RoundedRectangleBorder(
                          borderRadius:
                              BorderRadius
                                  .circular(
                            18,
                          ),
                        ),

                        child:
                            Padding(
                          padding:
                              const EdgeInsets
                                  .all(16),

                          child: Column(
                            crossAxisAlignment:
                                CrossAxisAlignment
                                    .start,

                            children: [
                              // =================================================
                              // TITLE
                              // =================================================

                              Row(
                                children: [
                                  Container(
                                    padding:
                                        const EdgeInsets
                                            .all(
                                      10,
                                    ),
                                    decoration:
                                        BoxDecoration(
                                      color: Colors
                                          .green
                                          .shade50,
                                      shape:
                                          BoxShape
                                              .circle,
                                    ),
                                    child:
                                        const Icon(
                                      Icons
                                          .agriculture,
                                      color:
                                          Colors
                                              .green,
                                    ),
                                  ),

                                  const SizedBox(
                                    width: 12,
                                  ),

                                  Expanded(
                                    child:
                                        Text(
                                      scheme[
                                              'title_kn'] ??
                                          scheme[
                                              'title'] ??
                                          'Government Scheme',
                                      style:
                                          const TextStyle(
                                        fontSize:
                                            18,
                                        fontWeight:
                                            FontWeight
                                                .bold,
                                      ),
                                    ),
                                  ),
                                ],
                              ),

                              const SizedBox(
                                height: 12,
                              ),

                              // =================================================
                              // DESCRIPTION
                              // =================================================

                              Text(
                                scheme[
                                        'description'] ??
                                    '',
                                style:
                                    const TextStyle(
                                  fontSize:
                                      15,
                                  height:
                                      1.4,
                                ),
                              ),

                              const SizedBox(
                                height: 12,
                              ),

                              // =================================================
                              // BENEFIT
                              // =================================================

                              const Text(
                                '💰 Benefit',
                                style:
                                    TextStyle(
                                  fontWeight:
                                      FontWeight
                                          .bold,
                                ),
                              ),

                              const SizedBox(
                                height: 4,
                              ),

                              Text(
                                scheme[
                                        'benefit'] ??
                                    '',
                                style:
                                    const TextStyle(
                                  height:
                                      1.4,
                                ),
                              ),

                              const SizedBox(
                                height: 12,
                              ),

                              // =================================================
                              // ELIGIBILITY
                              // =================================================

                              const Text(
                                '👨‍🌾 Eligibility',
                                style:
                                    TextStyle(
                                  fontWeight:
                                      FontWeight
                                          .bold,
                                ),
                              ),

                              const SizedBox(
                                height: 4,
                              ),

                              Text(
                                scheme[
                                        'eligibility'] ??
                                    '',
                                style:
                                    const TextStyle(
                                  height:
                                      1.4,
                                ),
                              ),

                              const SizedBox(
                                height: 14,
                              ),

                              // =================================================
                              // CATEGORY
                              // =================================================

                              Container(
                                padding:
                                    const EdgeInsets
                                        .symmetric(
                                  horizontal:
                                      10,
                                  vertical:
                                      6,
                                ),
                                decoration:
                                    BoxDecoration(
                                  color: Colors
                                      .green
                                      .shade50,
                                  borderRadius:
                                      BorderRadius
                                          .circular(
                                    20,
                                  ),
                                ),
                                child:
                                    Text(
                                  scheme[
                                          'category'] ??
                                      'Government Scheme',
                                  style:
                                      const TextStyle(
                                    color:
                                        Colors
                                            .green,
                                    fontWeight:
                                        FontWeight
                                            .bold,
                                  ),
                                ),
                              ),

                              const SizedBox(
                                height: 16,
                              ),

                              // =================================================
                              // OFFICIAL WEBSITE
                              // =================================================

                              SizedBox(
                                width:
                                    double.infinity,
                                child:
                                    ElevatedButton
                                        .icon(
                                  onPressed:
                                      () {
                                    openOfficialWebsite(
                                      scheme[
                                          'official_url'],
                                    );
                                  },
                                  icon:
                                      const Icon(
                                    Icons
                                        .open_in_new,
                                  ),
                                  label:
                                      const Text(
                                    'Official Website / Apply Now',
                                  ),
                                  style:
                                      ElevatedButton
                                          .styleFrom(
                                    backgroundColor:
                                        Colors
                                            .green,
                                    foregroundColor:
                                        Colors
                                            .white,
                                    padding:
                                        const EdgeInsets
                                            .symmetric(
                                      vertical:
                                          13,
                                    ),
                                    shape:
                                        RoundedRectangleBorder(
                                      borderRadius:
                                          BorderRadius
                                              .circular(
                                        12,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
    );
  }
}

// =========================================================
// AGRICULTURE MEDICINE PAGE
// =========================================================

class AgricultureMedicinePage
    extends StatefulWidget {
  const AgricultureMedicinePage({
    super.key,
  });

  @override
  State<AgricultureMedicinePage>
      createState() =>
          _AgricultureMedicinePageState();
}

class _AgricultureMedicinePageState
    extends State<
        AgricultureMedicinePage> {
  static const String backendUrl =
      'http://192.168.1.39:8000';

  bool isLoading = true;

  List<dynamic> medicines = [];

  // =========================================================
  // INITIALIZATION
  // =========================================================

  @override
  void initState() {
    super.initState();
    fetchMedicines();
  }

  // =========================================================
  // FETCH MEDICINES
  // =========================================================

  Future<void> fetchMedicines() async {
    try {
      final response = await http.get(
        Uri.parse(
          '$backendUrl/medicines',
        ),
      );

      if (!mounted) {
        return;
      }

      if (response.statusCode == 200) {
        final data =
            jsonDecode(response.body);

        setState(() {
          medicines =
              data['medicines'] ?? [];

          isLoading = false;
        });
      } else {
        setState(() {
          isLoading = false;
        });
      }
    } catch (e) {
      debugPrint(
        'Medicine Error: $e',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        isLoading = false;
      });
    }
  }

  // =========================================================
  // BUILD
  // =========================================================

  @override
  Widget build(
    BuildContext context,
  ) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'ಕೃಷಿ ಔಷಧಿ ಮಾಹಿತಿ 💊',
        ),
        backgroundColor:
            Colors.green,
        foregroundColor:
            Colors.white,
      ),

      body: isLoading
          ? const Center(
              child:
                  CircularProgressIndicator(),
            )
          : medicines.isEmpty
              ? const Center(
                  child: Text(
                    'ಯಾವುದೇ ಔಷಧಿ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.',
                  ),
                )
              : RefreshIndicator(
                  onRefresh:
                      fetchMedicines,

                  child:
                      ListView.builder(
                    padding:
                        const EdgeInsets.all(
                      16,
                    ),

                    itemCount:
                        medicines.length,

                    itemBuilder:
                        (context, index) {
                      final medicine =
                          medicines[index];

                      return Card(
                        margin:
                            const EdgeInsets
                                .only(
                          bottom: 16,
                        ),

                        elevation: 3,

                        shape:
                            RoundedRectangleBorder(
                          borderRadius:
                              BorderRadius
                                  .circular(
                            18,
                          ),
                        ),

                        child:
                            Padding(
                          padding:
                              const EdgeInsets
                                  .all(16),

                          child: Column(
                            crossAxisAlignment:
                                CrossAxisAlignment
                                    .start,

                            children: [
                              // =================================================
                              // CROP
                              // =================================================

                              Row(
                                children: [
                                  Container(
                                    padding:
                                        const EdgeInsets
                                            .all(
                                      10,
                                    ),
                                    decoration:
                                        BoxDecoration(
                                      color: Colors
                                          .green
                                          .shade50,
                                      shape:
                                          BoxShape
                                              .circle,
                                    ),
                                    child:
                                        const Icon(
                                      Icons
                                          .medication,
                                      color:
                                          Colors
                                              .green,
                                    ),
                                  ),

                                  const SizedBox(
                                    width: 12,
                                  ),

                                  Expanded(
                                    child:
                                        Text(
                                      medicine[
                                              'crop'] ??
                                          'ಬೆಳೆ',
                                      style:
                                          const TextStyle(
                                        fontSize:
                                            19,
                                        fontWeight:
                                            FontWeight
                                                .bold,
                                      ),
                                    ),
                                  ),
                                ],
                              ),

                              const SizedBox(
                                height: 14,
                              ),

                              // =================================================
                              // PROBLEM
                              // =================================================

                              Text(
                                '🐛 ಸಮಸ್ಯೆ: ${medicine['problem_kn'] ?? medicine['problem'] ?? ''}',
                                style:
                                    const TextStyle(
                                  fontSize:
                                      16,
                                  fontWeight:
                                      FontWeight
                                          .bold,
                                ),
                              ),

                              const SizedBox(
                                height: 10,
                              ),

                              // =================================================
                              // MEDICINE TYPE
                              // =================================================

                              Text(
                                '💊 ಔಷಧಿ ಪ್ರಕಾರ: ${medicine['medicine_type'] ?? ''}',
                                style:
                                    const TextStyle(
                                  fontSize:
                                      15,
                                ),
                              ),

                              const SizedBox(
                                height: 12,
                              ),

                              // =================================================
                              // INFORMATION
                              // =================================================

                              const Text(
                                '📋 ಮಾಹಿತಿ',
                                style:
                                    TextStyle(
                                  fontWeight:
                                      FontWeight
                                          .bold,
                                  fontSize:
                                      16,
                                ),
                              ),

                              const SizedBox(
                                height: 5,
                              ),

                              Text(
                                medicine[
                                        'general_info'] ??
                                    '',
                                style:
                                    const TextStyle(
                                  fontSize:
                                      14.5,
                                  height:
                                      1.5,
                                ),
                              ),

                              const SizedBox(
                                height: 12,
                              ),

                              // =================================================
                              // USAGE
                              // =================================================

                              const Text(
                                '🧑‍🌾 ಬಳಕೆ',
                                style:
                                    TextStyle(
                                  fontWeight:
                                      FontWeight
                                          .bold,
                                  fontSize:
                                      16,
                                ),
                              ),

                              const SizedBox(
                                height: 5,
                              ),

                              Text(
                                medicine[
                                        'usage'] ??
                                    '',
                                style:
                                    const TextStyle(
                                  fontSize:
                                      14.5,
                                  height:
                                      1.5,
                                ),
                              ),

                              const SizedBox(
                                height: 12,
                              ),

                              // =================================================
                              // PRECAUTION
                              // =================================================

                              Container(
                                width:
                                    double.infinity,
                                padding:
                                    const EdgeInsets
                                        .all(
                                  12,
                                ),
                                decoration:
                                    BoxDecoration(
                                  color: Colors
                                      .orange
                                      .shade50,
                                  borderRadius:
                                      BorderRadius
                                          .circular(
                                    12,
                                  ),
                                  border:
                                      Border.all(
                                    color: Colors
                                        .orange
                                        .shade100,
                                  ),
                                ),
                                child:
                                    Text(
                                  '⚠️ ಮುನ್ನೆಚ್ಚರಿಕೆ\n${medicine['precaution'] ?? ''}',
                                  style:
                                      const TextStyle(
                                    fontSize:
                                        14,
                                    height:
                                        1.45,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
    );
  }
}


class FarmerNotificationsPage extends StatefulWidget {
  const FarmerNotificationsPage({super.key});

  @override
  State<FarmerNotificationsPage> createState() =>
      _FarmerNotificationsPageState();
}

class _FarmerNotificationsPageState
    extends State<FarmerNotificationsPage> {
  static const String backendUrl = 'http://192.168.1.39:8000';

  bool isLoading = true;
  List<dynamic> notifications = [];

  @override
  void initState() {
    super.initState();
    fetchNotifications();
  }

  Future<void> fetchNotifications() async {
    try {
      final response = await http.get(
        Uri.parse('$backendUrl/notifications'),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        setState(() {
          notifications = data['notifications'] ?? [];
          isLoading = false;
        });
      } else {
        setState(() {
          isLoading = false;
        });
      }
    } catch (e) {
      debugPrint('Notifications Error: $e');

      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Farmer Notifications',
          style: TextStyle(
            fontWeight: FontWeight.bold,
          ),
        ),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: isLoading
          ? const Center(
              child: CircularProgressIndicator(
                color: Colors.green,
              ),
            )
          : notifications.isEmpty
              ? const Center(
                  child: Text(
                    'ಯಾವುದೇ notifications ಇಲ್ಲ.',
                    style: TextStyle(fontSize: 18),
                  ),
                )
              : RefreshIndicator(
                  color: Colors.green,
                  onRefresh: fetchNotifications,
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: notifications.length,
                    itemBuilder: (context, index) {
                      final notification = notifications[index];

                      final title =
                          notification['title']?.toString() ?? '';

                      final message =
                          notification['message']?.toString() ?? '';

                      final category =
                          notification['category']?.toString() ?? '';

                      final date =
                          notification['date']?.toString() ?? '';

                      final isNew =
                          notification['is_new'] == true;

                      return Card(
                        elevation: 3,
                        margin: const EdgeInsets.only(bottom: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Row(
                            crossAxisAlignment:
                                CrossAxisAlignment.start,
                            children: [
                              const CircleAvatar(
                                backgroundColor: Colors.green,
                                child: Icon(
                                  Icons.notifications,
                                  color: Colors.white,
                                ),
                              ),

                              const SizedBox(width: 14),

                              Expanded(
                                child: Column(
                                  crossAxisAlignment:
                                      CrossAxisAlignment.start,
                                  children: [
                                    Row(
                                      children: [
                                        Expanded(
                                          child: Text(
                                            title,
                                            style: const TextStyle(
                                              fontSize: 17,
                                              fontWeight:
                                                  FontWeight.bold,
                                            ),
                                          ),
                                        ),

                                        if (isNew)
                                          Container(
                                            padding:
                                                const EdgeInsets
                                                    .symmetric(
                                              horizontal: 8,
                                              vertical: 4,
                                            ),
                                            decoration:
                                                BoxDecoration(
                                              color: Colors.red,
                                              borderRadius:
                                                  BorderRadius.circular(
                                                      12),
                                            ),
                                            child: const Text(
                                              'NEW',
                                              style: TextStyle(
                                                color: Colors.white,
                                                fontSize: 11,
                                                fontWeight:
                                                    FontWeight.bold,
                                              ),
                                            ),
                                          ),
                                      ],
                                    ),

                                    const SizedBox(height: 8),

                                    Text(
                                      message,
                                      style: const TextStyle(
                                        fontSize: 14,
                                        height: 1.4,
                                      ),
                                    ),

                                    const SizedBox(height: 10),

                                    Text(
                                      '$category • $date',
                                      style: const TextStyle(
                                        color: Colors.green,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
    );
  }
}