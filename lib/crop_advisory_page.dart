import 'package:flutter/material.dart';

class CropAdvisoryPage extends StatefulWidget {
  const CropAdvisoryPage({super.key});
  @override
  State<CropAdvisoryPage> createState() => _CropAdvisoryPageState();
}

class _CropAdvisoryPageState extends State<CropAdvisoryPage> {
  String? selectedCrop;
  String advice = '';

  final List<String> crops = [
    'ಅಕ್ಕಿ','ರಾಗಿ','ಮೆಕ್ಕೆಜೋಳ','ಗೋಧಿ','ಜೋಳ','ಸಜ್ಜೆ','ತೊಗರಿ','ಹೆಸರುಕಾಳು','ಉದ್ದು','ಅವರೆ','ಕಡಲೆ','ಸೋಯಾಬೀನ್','ಕಡಲೆಕಾಯಿ','ಎಳ್ಳು','ಸೂರ್ಯಕಾಂತಿ','ಹತ್ತಿ','ಕಬ್ಬು','ತಂಬಾಕು',
    'ಟೊಮೇಟೊ','ಈರುಳ್ಳಿ','ಆಲೂಗಡ್ಡೆ','ಮೆಣಸಿನಕಾಯಿ','ಬದನೆಕಾಯಿ','ಬೆಂಡೆಕಾಯಿ','ಕ್ಯಾರೆಟ್','ಮೂಲಂಗಿ','ಎಲೆಕೋಸು','ಹೂಕೋಸು','ಸೌತೆಕಾಯಿ','ಕುಂಬಳಕಾಯಿ','ಹಾಗಲಕಾಯಿ','ಸೀಮೆ ಬದನೆ',
    'ಬಾಳೆ','ಮಾವು','ಪೇರಳೆ','ಸಪೋಟಾ','ದಾಳಿಂಬೆ','ಪಪ್ಪಾಯಿ','ಕಲ್ಲಂಗಡಿ','ಅನಾನಸ್','ದ್ರಾಕ್ಷಿ','ಕಿತ್ತಳೆ','ತೆಂಗು','ಅಡಿಕೆ','ಕಾಫಿ','ಚಹಾ','ಕಾಳುಮೆಣಸು','ಏಲಕ್ಕಿ','ಶುಂಠಿ','ಅರಿಶಿನ','ವೆನಿಲ್ಲಾ','ರಬ್ಬರ್','ಕೊಕೊ','ಹೂ ಬೆಳೆಗಳು',
  ];

  final Map<String, String> cropAdvice = {
    'ಅಕ್ಕಿ': '🌾 ಅಕ್ಕಿ: ನೀರಿನ ನಿರ್ವಹಣೆ ಸರಿಯಾಗಿ ಮಾಡಿ. ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಆಧಾರದ ಮೇಲೆ ಗೊಬ್ಬರ ಬಳಸಿ. ಕಾಂಡಕೊರೆಯುವ ಕೀಟ ಮತ್ತು ಎಲೆ ರೋಗಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.',
    'ರಾಗಿ': '🌱 ರಾಗಿ: ಉತ್ತಮ ಬೀಜ ಬಳಸಿ. ಆರಂಭಿಕ ಹಂತದಲ್ಲಿ ಕಳೆ ನಿಯಂತ್ರಿಸಿ. ಮಣ್ಣಿನ ತೇವಾಂಶ ಕಾಪಾಡಿ ಮತ್ತು ಸಮತೋಲನ ಗೊಬ್ಬರ ಬಳಸಿ.',
    'ಮೆಕ್ಕೆಜೋಳ': '🌽 ಮೆಕ್ಕೆಜೋಳ: ಉತ್ತಮ ಬೀಜ ಬಳಸಿ. ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. ಎಲೆ ಮತ್ತು ಕಾಂಡದ ಕೀಟಗಳನ್ನು ಗಮನಿಸಿ.',
    'ಗೋಧಿ': '🌾 ಗೋಧಿ: ಸರಿಯಾದ ಸಮಯದಲ್ಲಿ ಬಿತ್ತನೆ ಮಾಡಿ. ಅಗತ್ಯಕ್ಕೆ ತಕ್ಕಂತೆ ನೀರು ನೀಡಿ. ತುಕ್ಕು ರೋಗ ಮತ್ತು ಕೀಟಗಳ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.',
    'ಜೋಳ': '🌾 ಜೋಳ: ಮಣ್ಣಿನ ತೇವಾಂಶ ಕಾಪಾಡಿ. ಕಳೆ ನಿಯಂತ್ರಿಸಿ. ಕಾಂಡಕೊರೆಯುವ ಕೀಟದ ಹಾನಿಯನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಸಜ್ಜೆ': '🌾 ಸಜ್ಜೆ: ಬರ ಪರಿಸ್ಥಿತಿಗೆ ತಕ್ಕ ನಿರ್ವಹಣೆ ಮಾಡಿ. ಉತ್ತಮ ಬೀಜ ಬಳಸಿ ಮತ್ತು ಆರಂಭಿಕ ಹಂತದಲ್ಲಿ ಕಳೆ ನಿಯಂತ್ರಿಸಿ.',
    'ತೊಗರಿ': '🫘 ತೊಗರಿ: ನೀರು ನಿಲ್ಲದ ಮಣ್ಣು ಉತ್ತಮ. ಕಾಯಿ ಕೊರೆಯುವ ಕೀಟ ಮತ್ತು wilt ರೋಗವನ್ನು ಗಮನಿಸಿ. ಸಮತೋಲನ ಗೊಬ್ಬರ ಬಳಸಿ.',
    'ಹೆಸರುಕಾಳು': '🫘 ಹೆಸರುಕಾಳು: ಉತ್ತಮ ಗುಣಮಟ್ಟದ ಬೀಜ ಬಳಸಿ. ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. ಕೀಟ ಮತ್ತು ಎಲೆ ರೋಗಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಉದ್ದು': '🫘 ಉದ್ದು: ಮಣ್ಣಿನ ತೇವಾಂಶ ಕಾಪಾಡಿ. ಕಳೆ ನಿಯಂತ್ರಿಸಿ. ಎಲೆ ಕಲೆ ಮತ್ತು ಕೀಟಗಳ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.',
    'ಅವರೆ': '🫘 ಅವರೆ: ಬಳ್ಳಿಗೆ ಆಧಾರ ಒದಗಿಸಿ. ನಿಯಮಿತವಾಗಿ ನೀರು ನೀಡಿ. ಕಾಯಿ ಮತ್ತು ಎಲೆಗಳಲ್ಲಿ ಕೀಟ ಹಾನಿ ಪರಿಶೀಲಿಸಿ.',
    'ಕಡಲೆ': '🫘 ಕಡಲೆ: ನೀರು ನಿಲ್ಲದ ಮಣ್ಣು ಬಳಸಿ. ಹೂಬಿಡುವ ಹಂತದಲ್ಲಿ ಅಗತ್ಯ ನೀರು ನೀಡಿ. pod borer ಕೀಟವನ್ನು ಗಮನಿಸಿ.',
    'ಸೋಯಾಬೀನ್': '🫘 ಸೋಯಾಬೀನ್: ಉತ್ತಮ ಬೀಜ ಮತ್ತು ಸರಿಯಾದ ಅಂತರದಲ್ಲಿ ಬಿತ್ತನೆ ಮಾಡಿ. ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. ಎಲೆ ತಿನ್ನುವ ಕೀಟಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಕಡಲೆಕಾಯಿ': '🥜 ಕಡಲೆಕಾಯಿ: ಹಗುರವಾದ, ನೀರು ಹರಿಯುವ ಮಣ್ಣು ಉತ್ತಮ. ಹೂಬಿಡುವ ಹಂತದಲ್ಲಿ ತೇವಾಂಶ ಕಾಪಾಡಿ. ಎಲೆ ಕಲೆ ಮತ್ತು ಕೀಟಗಳನ್ನು ಗಮನಿಸಿ.',
    'ಎಳ್ಳು': '🌱 ಎಳ್ಳು: ನೀರು ನಿಲ್ಲದ ಮಣ್ಣು ಅಗತ್ಯ. ಕಳೆ ನಿಯಂತ್ರಿಸಿ. ಎಲೆ ಮತ್ತು ಕಾಯಿಗಳ ಮೇಲೆ ಕೀಟ ಹಾನಿ ಪರಿಶೀಲಿಸಿ.',
    'ಸೂರ್ಯಕಾಂತಿ': '🌻 ಸೂರ್ಯಕಾಂತಿ: ಆರಂಭಿಕ ಹಂತದಲ್ಲಿ ಕಳೆ ನಿಯಂತ್ರಿಸಿ. ಅಗತ್ಯ ನೀರು ನೀಡಿ. ತಲೆ ಮತ್ತು ಎಲೆಗಳ ಕೀಟಗಳನ್ನು ಗಮನಿಸಿ.',
    'ಹತ್ತಿ': '🌿 ಹತ್ತಿ: ನಿಯಮಿತವಾಗಿ ಹೊಲ ಪರಿಶೀಲಿಸಿ. bollworm ಮತ್ತು sucking pests ಗಮನಿಸಿ. ಗೊಬ್ಬರವನ್ನು ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಆಧಾರದ ಮೇಲೆ ಬಳಸಿ.',
    'ಕಬ್ಬು': '🎋 ಕಬ್ಬು: ಸಮರ್ಪಕ ನೀರು ಮತ್ತು ಪೋಷಕಾಂಶ ನೀಡಿ. ಕಳೆ ನಿಯಂತ್ರಿಸಿ. ಕಾಂಡ ಕೊರೆಯುವ ಕೀಟ ಮತ್ತು ರೋಗಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ತಂಬಾಕು': '🌿 ತಂಬಾಕು: ಮಣ್ಣಿನ ತೇವಾಂಶ ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ. ಎಲೆ ಕೀಟ ಮತ್ತು ರೋಗಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.',
    'ಟೊಮೇಟೊ': '🍅 ಟೊಮೇಟೊ: ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. support ಒದಗಿಸಿ. fruit borer ಮತ್ತು leaf spot ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಈರುಳ್ಳಿ': '🧅 ಈರುಳ್ಳಿ: ಮಣ್ಣಿನಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. ಸಮತೋಲನ ಗೊಬ್ಬರ ಬಳಸಿ. thrips ಮತ್ತು purple blotch ರೋಗವನ್ನು ಗಮನಿಸಿ.',
    'ಆಲೂಗಡ್ಡೆ': '🥔 ಆಲೂಗಡ್ಡೆ: ಉತ್ತಮ drainage ಅಗತ್ಯ. earthing up ಮಾಡಿ. late blight ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.',
    'ಮೆಣಸಿನಕಾಯಿ': '🌶️ ಮೆಣಸಿನಕಾಯಿ: ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. thrips, mites ಮತ್ತು fruit rot ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಬದನೆಕಾಯಿ': '🍆 ಬದನೆಕಾಯಿ: ನಿಯಮಿತವಾಗಿ ನೀರು ನೀಡಿ. shoot and fruit borer ಕೀಟವನ್ನು ಗಮನಿಸಿ. ಹಾನಿಗೊಳಗಾದ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.',
    'ಬೆಂಡೆಕಾಯಿ': '🌿 ಬೆಂಡೆಕಾಯಿ: ನಿಯಮಿತ ಕೊಯ್ಲು ಮಾಡಿ. fruit borer ಮತ್ತು sucking pests ಪರಿಶೀಲಿಸಿ. ಹೊಲ ಸ್ವಚ್ಛವಾಗಿಡಿ.',
    'ಕ್ಯಾರೆಟ್': '🥕 ಕ್ಯಾರೆಟ್: ಸಡಿಲವಾದ ಮಣ್ಣು ಉತ್ತಮ. ಸಮರ್ಪಕ ತೇವಾಂಶ ಕಾಪಾಡಿ. ಬೇರು ಹಾನಿ ಮತ್ತು ಕೀಟಗಳನ್ನು ಗಮನಿಸಿ.',
    'ಮೂಲಂಗಿ': '🌱 ಮೂಲಂಗಿ: ಸಡಿಲವಾದ, ನೀರು ಹರಿಯುವ ಮಣ್ಣು ಬಳಸಿ. ಸಮತೋಲನ ನೀರು ನೀಡಿ. ಬೇರು ಕೀಟಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಎಲೆಕೋಸು': '🥬 ಎಲೆಕೋಸು: ತೇವಾಂಶ ಕಾಪಾಡಿ. cabbage worm ಮತ್ತು aphids ಗಮನಿಸಿ. ಹಾನಿಗೊಳಗಾದ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.',
    'ಹೂಕೋಸು': '🥦 ಹೂಕೋಸು: ಸಮರ್ಪಕ ನೀರು ಮತ್ತು ಪೋಷಕಾಂಶ ನೀಡಿ. diamondback moth ಮತ್ತು rot ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಸೌತೆಕಾಯಿ': '🥒 ಸೌತೆಕಾಯಿ: ಬಳ್ಳಿಗೆ support ಒದಗಿಸಿ. ನಿಯಮಿತ ನೀರು ನೀಡಿ. powdery mildew ಮತ್ತು fruit fly ಗಮನಿಸಿ.',
    'ಕುಂಬಳಕಾಯಿ': '🎃 ಕುಂಬಳಕಾಯಿ: ಉತ್ತಮ drainage ಇರುವ ಮಣ್ಣು ಬಳಸಿ. ಬಳ್ಳಿಗೆ ಸಾಕಷ್ಟು ಜಾಗ ನೀಡಿ. fruit fly ಮತ್ತು mildew ಗಮನಿಸಿ.',
    'ಹಾಗಲಕಾಯಿ': '🥒 ಹಾಗಲಕಾಯಿ: ಬಳ್ಳಿಗೆ support ನೀಡಿ. ನಿಯಮಿತವಾಗಿ ನೀರು ನೀಡಿ. fruit fly ಮತ್ತು powdery mildew ಪರಿಶೀಲಿಸಿ.',
    'ಸೀಮೆ ಬದನೆ': '🌿 ಸೀಮೆ ಬದನೆ: ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. ಕೀಟ ಮತ್ತು ಎಲೆ ರೋಗಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ. ಅಗತ್ಯ support ನೀಡಿ.',
    'ಬಾಳೆ': '🍌 ಬಾಳೆ: ನಿಯಮಿತ ನೀರು ಮತ್ತು ಪೋಷಕಾಂಶ ನೀಡಿ. ಗಾಳಿಯಿಂದ ಗಿಡ ಬೀಳದಂತೆ support ನೀಡಿ. pseudostem borer ಮತ್ತು sigatoka ಗಮನಿಸಿ.',
    'ಮಾವು': '🥭 ಮಾವು: ತೋಟದಲ್ಲಿ ಸ್ವಚ್ಛತೆ ಕಾಪಾಡಿ. ಹೂ ಮತ್ತು ಹಣ್ಣು ಹಂತದಲ್ಲಿ ನೀರಿನ ನಿರ್ವಹಣೆ ಮಾಡಿ. fruit fly ಮತ್ತು powdery mildew ಗಮನಿಸಿ.',
    'ಪೇರಳೆ': '🍐 ಪೇರಳೆ: ಸಮರ್ಪಕ ನೀರು ನೀಡಿ. ಗಿಡದ ಸುತ್ತ ಸ್ವಚ್ಛತೆ ಕಾಪಾಡಿ. fruit fly ಮತ್ತು leaf spot ಗಮನಿಸಿ.',
    'ಸಪೋಟಾ': '🍈 ಸಪೋಟಾ: ಉತ್ತಮ drainage ಮತ್ತು ಸಮತೋಲನ ಗೊಬ್ಬರ ಬಳಸಿ. ಹಣ್ಣು ಕೀಟ ಮತ್ತು ಎಲೆ ರೋಗಗಳನ್ನು ಗಮನಿಸಿ.',
    'ದಾಳಿಂಬೆ': '❤️ ದಾಳಿಂಬೆ: ನೀರಿನ ನಿರ್ವಹಣೆ ಸಮರ್ಪಕವಾಗಿರಲಿ. fruit borer ಮತ್ತು bacterial blight ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ. ಒಣ ಕೊಂಬೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.',
    'ಪಪ್ಪಾಯಿ': '🥭 ಪಪ್ಪಾಯಿ: ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. mealybug ಮತ್ತು viral disease ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ. ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ಪ್ರತ್ಯೇಕಿಸಿ.',
    'ಕಲ್ಲಂಗಡಿ': '🍉 ಕಲ್ಲಂಗಡಿ: ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. fruit fly ಮತ್ತು powdery mildew ಗಮನಿಸಿ. ಹಣ್ಣು ಬೆಳೆಯುವ ಹಂತದಲ್ಲಿ ನೀರಿನ ನಿಯಂತ್ರಣ ಮಾಡಿ.',
    'ಅನಾನಸ್': '🍍 ಅನಾನಸ್: ಮಣ್ಣಿನಲ್ಲಿ ಉತ್ತಮ drainage ಇರಲಿ. ಸಮತೋಲನ ಪೋಷಕಾಂಶ ನೀಡಿ. mealybug ಮತ್ತು rot ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.',
    'ದ್ರಾಕ್ಷಿ': '🍇 ದ್ರಾಕ್ಷಿ: pruning ಮತ್ತು canopy management ಸರಿಯಾಗಿ ಮಾಡಿ. powdery mildew, downy mildew ಮತ್ತು ಹಣ್ಣು ಕೀಟಗಳನ್ನು ಗಮನಿಸಿ.',
    'ಕಿತ್ತಳೆ': '🍊 ಕಿತ್ತಳೆ: ನಿಯಮಿತ ನೀರು ಮತ್ತು ಪೋಷಕಾಂಶ ನೀಡಿ. citrus psylla ಮತ್ತು canker ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ತೆಂಗು': '🥥 ತೆಂಗು: ಬೇಸಿಗೆಯಲ್ಲಿ ಸಮರ್ಪಕ ನೀರು ನೀಡಿ. ತೋಟ ಸ್ವಚ್ಛವಾಗಿಡಿ. rhinoceros beetle ಮತ್ತು red palm weevil ಗಮನಿಸಿ.',
    'ಅಡಿಕೆ': '🌴 ಅಡಿಕೆ: ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. ಸಮತೋಲನ ಗೊಬ್ಬರ ಬಳಸಿ. fruit rot ಮತ್ತು spindle bug ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಕಾಫಿ': '☕ ಕಾಫಿ: ನೆರಳು ಮತ್ತು ತೇವಾಂಶ ಸಮತೋಲನ ಕಾಪಾಡಿ. coffee berry borer ಮತ್ತು leaf rust ಗಮನಿಸಿ. ಒಣ ಕೊಂಬೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.',
    'ಚಹಾ': '🍃 ಚಹಾ: ಸಮರ್ಪಕ ತೇವಾಂಶ ಮತ್ತು pruning ಕಾಪಾಡಿ. tea mosquito bug ಮತ್ತು blister blight ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಕಾಳುಮೆಣಸು': '🌿 ಕಾಳುಮೆಣಸು: ಬಳ್ಳಿಗೆ ಉತ್ತಮ support ನೀಡಿ. ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. quick wilt ಮತ್ತು pollu beetle ಗಮನಿಸಿ.',
    'ಏಲಕ್ಕಿ': '🌿 ಏಲಕ್ಕಿ: ತೇವಾಂಶ ಮತ್ತು ನೆರಳು ಸಮತೋಲನ ಕಾಪಾಡಿ. capsule rot ಮತ್ತು shoot borer ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಶುಂಠಿ': '🫚 ಶುಂಠಿ: ಉತ್ತಮ drainage ಇರುವ ಮಣ್ಣು ಬಳಸಿ. ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ. rhizome rot ಮತ್ತು shoot borer ಗಮನಿಸಿ.',
    'ಅರಿಶಿನ': '🌿 ಅರಿಶಿನ: ಮಣ್ಣಿನ ತೇವಾಂಶ ಕಾಪಾಡಿ. ಉತ್ತಮ drainage ಇರಲಿ. rhizome rot ಮತ್ತು leaf spot ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ವೆನಿಲ್ಲಾ': '🌿 ವೆನಿಲ್ಲಾ: support ಮರ ಮತ್ತು ನೆರಳು ಕಾಪಾಡಿ. ಮಿತವಾದ ತೇವಾಂಶ ಇರಲಿ. stem rot ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.',
    'ರಬ್ಬರ್': '🌳 ರಬ್ಬರ್: ತೋಟದಲ್ಲಿ drainage ಕಾಪಾಡಿ. tapping ಅನ್ನು ಸರಿಯಾದ ವಿಧಾನದಲ್ಲಿ ಮಾಡಿ. leaf fall ಮತ್ತು panel disease ಗಮನಿಸಿ.',
    'ಕೊಕೊ': '🍫 ಕೊಕೊ: ನೆರಳು ಮತ್ತು ತೇವಾಂಶ ಸಮತೋಲನ ಕಾಪಾಡಿ. pod borer ಮತ್ತು black pod ರೋಗವನ್ನು ಪರಿಶೀಲಿಸಿ.',
    'ಹೂ ಬೆಳೆಗಳು': '🌸 ಹೂ ಬೆಳೆಗಳು: ಉತ್ತಮ drainage ಮತ್ತು ಸಮತೋಲನ ಪೋಷಕಾಂಶ ನೀಡಿ. ಕೀಟ ಮತ್ತು ಶಿಲೀಂಧ್ರ ರೋಗಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.',
  };

  void getAdvice() {
    setState(() {
      advice = selectedCrop == null
          ? 'ದಯವಿಟ್ಟು ಮೊದಲು ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ.'
          : cropAdvice[selectedCrop!] ?? '🌱 ${selectedCrop!}: ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಆಧಾರದ ಮೇಲೆ ಗೊಬ್ಬರ ಬಳಸಿ. ಸಮರ್ಪಕ ನೀರು ನೀಡಿ. ಕೀಟ ಮತ್ತು ರೋಗಗಳಿಗಾಗಿ ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ. ಗಂಭೀರ ಸಮಸ್ಯೆ ಕಂಡುಬಂದರೆ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ.';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ಬೆಳೆ ಸಲಹೆ', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            elevation: 4,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  const Icon(Icons.agriculture, size: 70, color: Colors.green),
                  const SizedBox(height: 12),
                  const Text('ನಿಮ್ಮ ಬೆಳೆಗೆ ಸಲಹೆ', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  const Text('ನಿಮ್ಮ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ ಮತ್ತು ಕೃಷಿ ಸಲಹೆ ಪಡೆಯಿರಿ', textAlign: TextAlign.center, style: TextStyle(fontSize: 15)),
                  const SizedBox(height: 20),
                  DropdownButtonFormField<String>(
                    initialValue: selectedCrop,
                    isExpanded: true,
                    decoration: InputDecoration(
                      labelText: 'ಬೆಳೆ ಆಯ್ಕೆಮಾಡಿ',
                      prefixIcon: const Icon(Icons.grass, color: Colors.green),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                    items: crops.map((crop) => DropdownMenuItem<String>(value: crop, child: Text(crop))).toList(),
                    onChanged: (value) => setState(() { selectedCrop = value; advice = ''; }),
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: getAdvice,
                      icon: const Icon(Icons.search),
                      label: const Text('ಸಲಹೆ ಪಡೆಯಿರಿ', style: TextStyle(fontSize: 16)),
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.green, foregroundColor: Colors.white, padding: const EdgeInsets.symmetric(vertical: 14)),
                    ),
                  ),
                  if (advice.isNotEmpty) ...[
                    const SizedBox(height: 20),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(color: Colors.green.shade50, borderRadius: BorderRadius.circular(16), border: Border.all(color: Colors.green)),
                      child: Text(advice, style: const TextStyle(fontSize: 16, height: 1.5)),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
