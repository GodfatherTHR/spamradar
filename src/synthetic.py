from __future__ import annotations

import random
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Iterable

ZERO_WIDTH = "\u200b"

SCAM_OPENERS: dict[str, list[str]] = {
    "English": [
        "URGENT:", "Final notice:", "Congratulations!", "Dear customer,",
        "Attention:", "ALERT:", "Hi dear,", "Good news!", "Hello,",
        "Last warning:", "Important:",
    ],
    "Bengali": [
        "জরুরি:", "শেষ সুযোগ:", "অভিনন্দন!", "প্রিয় গ্রাহক,",
        "মনোযোগ দিন:", "সতর্কতা:", "প্রিয়,",
    ],
    "Banglish": [
        "Joruri:", "Sesh sujog:", "Obhinondon!", "Priyo grahok,",
        "Monojog din:", "Shobdhan:", "Priyo,",
    ],
    "Hindi": [
        "जरूरी:", "अंतिम मौका:", "बधाई हो!", "प्रिय ग्राहक,",
        "ध्यान दें:", "चेतावनी:", "प्रिय,",
    ],
    "Hinglish": [
        "Jaruri:", "Aakhri mauka:", "Badhai ho!", "Priya grahak,",
        "Dhyan dein:", "Warning:", "Priye,",
    ],
    "Spanish": [
        "URGENTE:", "Último aviso:", "¡Felicidades!", "Estimado cliente,",
        "Atención:", "ALERTA:", "Hola,",
    ],
    "French": [
        "URGENT :", "Dernier avis :", "Félicitations !", "Cher client,",
        "Attention :", "ALERTE :", "Bonjour,",
    ],
    "Arabic": [
        "عاجل:", "تنبيه أخير:", "تهانينا!", "عميلنا العزيز،",
        "انتبه:", "تحذير:", "مرحبا،",
    ],
}

SCAM_CTAS: dict[str, list[str]] = {
    "English": [
        "Click here to claim now: {url}", "Verify immediately at {url}",
        "Send your details to {wallet}", "Call us now at {phone}",
        "Reply YES to confirm", "Pay the fee at {url}",
        "Contact us on WhatsApp: {phone}", "Join our Telegram: {url}",
        "Confirm your account at {url}", "Send a message to {email}",
    ],
    "Bengali": [
        "এখনই দাবি করতে ক্লিক করুন: {url}", "দ্রুত যাচাই করুন: {url}",
        "আপনার তথ্য পাঠান: {wallet}", "এখনই কল করুন: {phone}",
        "হ্যাঁ লিখে উত্তর দিন", "ফি পরিশোধ করুন: {url}",
    ],
    "Banglish": [
        "Ekhoni claim korte click korun: {url}", "Turant verify korun: {url}",
        "Apnar details pathan: {wallet}", "Ekhoni call korun: {phone}",
        "Haan likhe reply din", "Fee payment korun: {url}",
    ],
    "Hindi": [
        "अभी दावा करने के लिए क्लिक करें: {url}", "तुरंत सत्यापित करें: {url}",
        "अपनी जानकारी भेजें: {wallet}", "अभी कॉल करें: {phone}",
        "हाँ लिखकर जवाब दें", "शुल्क का भुगतान करें: {url}",
    ],
    "Hinglish": [
        "Abhi claim karne ke liye click karein: {url}", "Turant verify karein: {url}",
        "Apni details bhejein: {wallet}", "Abhi call karein: {phone}",
        "Haan likhkar reply karein", "Fee pay karein: {url}",
    ],
    "Spanish": [
        "Haga clic aquí para reclamar: {url}", "Verifique de inmediato en {url}",
        "Envíe sus datos a {wallet}", "Llame ahora al {phone}",
        "Responda SÍ para confirmar", "Pague la tarifa en {url}",
    ],
    "French": [
        "Cliquez ici pour réclamer : {url}", "Vérifiez immédiatement sur {url}",
        "Envoyez vos informations à {wallet}", "Appelez-nous maintenant au {phone}",
        "Répondez OUI pour confirmer", "Payez les frais sur {url}",
    ],
    "Arabic": [
        "انقر هنا للمطالبة الآن: {url}", "تحقق فوراً من {url}",
        "أرسل بياناتك إلى {wallet}", "اتصل بنا الآن على {phone}",
        "رد بـ نعم للتأكيد", "ادفع الرسوم في {url}",
    ],
}

SCAM_BODIES: dict[str, dict[str, list[str]]] = {
    "English": {
        "p2p_bait": [
            "I need a small favor. Send me {amount} and I will send you a brand new {item} worth {amount2}. Only one left, bro.",
            "You pay {fee} for shipping and I will give you a {item}. I am serious, no scam.",
            "I accidentally sent you {amount2}. Send me {amount} back and keep the rest as a thank-you.",
        ],
        "crypto": [
            "Send {crypto} to this wallet and get {crypto2} back within 10 minutes. Guaranteed {pct} profit.",
            "Our AI trading bot doubles your {crypto} daily. Join the VIP Telegram group now: {url}",
            "Claim your {pct} crypto bonus. Verify your wallet at {url} before it expires.",
        ],
        "lottery_prize": [
            "Your number has won {prize} in the {brand} lottery. To claim, send your {wallet} number and a {fee} processing fee.",
            "You are today's lucky winner of a {item}. Claim within 24 hours at {url}.",
            "Congratulations! You have been selected for a {prize} cash prize. Confirm your details at {url}.",
        ],
        "bank_kyc": [
            "Your {bank} account is suspended due to suspicious activity. Verify your KYC immediately: {url}",
            "ATM card blocked. Update your details at {url} or your account will be closed within 24 hours.",
            "Security alert: an unauthorized login was detected on your {bank} account. Verify now at {url}.",
        ],
        "delivery_otp": [
            "Your parcel is held at customs. Pay a {fee} clearance fee here: {url}",
            "Delivery failed. Reschedule and pay {fee} at {url}. AWB {ref}.",
            "Our courier needs the OTP sent to your phone to complete delivery. Reply with the {otp} code.",
        ],
        "job": [
            "Work from home and earn {salary}. No experience needed. WhatsApp us at {phone}.",
            "Urgent hiring for data entry. Payout: {salary}. Join Telegram: {url}",
            "Part-time job, 2 hours a day, {salary}. No interview. Contact {phone}.",
        ],
        "romance": [
            "My dear, I am a {occupation} stationed overseas. I need {amount} for a visa to finally meet you. Please send it via {wallet}.",
            "I feel a deep connection with you. Can you help me with {amount} for my mother's hospital bill? I will repay you.",
            "Hello beautiful, I am a soldier on a peacekeeping mission. I want to send you a gift, but the customs fee is {fee}. Can you pay it?",
        ],
        "marketplace": [
            "Selling my {item} for only {amount}. I am deployed overseas, so I will ship it after you send a {fee} deposit via {wallet}.",
            "I have two {item} tickets. Zelle only. Pay a {amount} deposit now and I will transfer the tickets.",
            "Brand new {item}, sealed box, {amount}. I only accept Cash App. Pay first and I will ship today.",
        ],
        "loan_investment": [
            "Pre-approved loan of {amount} at 0% interest. No documents needed. Apply here: {url}",
            "Guaranteed {pct} returns in 7 days. Minimum investment {amount}. Join now: {url}",
            "Exclusive investment plan: double your {amount} in 24 hours. Limited slots. Contact {phone}.",
        ],
        "tech_support": [
            "Your device is infected with a virus. Call {phone} immediately and allow remote access to fix it.",
            "Microsoft security alert: your Windows license is expired. Renew at {url}.",
            "Your Apple ID has been locked. Call support at {phone} and read the verification code.",
        ],
        "promo": [
            "FLAT {pct} OFF today only! Use code {code} at {url}. Limited stock!",
            "Exclusive offer: {amount} cashback on your next recharge. Claim at {url}.",
            "Final sale: buy one get three free. Order now at {url} before midnight.",
        ],
    },
    "Bengali": {
        "p2p_bait": [
            "ভাই, আমি একটি {item} নিতে চাই। তুমি {amount} পাঠাও, আমি তোমাকে {amount2} ফেরত দেব।",
            "আমাকে {amount} পাঠান, আমি আপনাকে নতুন {item} উপহার দেব। মাত্র একটি বাকি।",
        ],
        "crypto": [
            "মাত্র {crypto} পাঠান, ১০ মিনিটে {crypto2} ফেরত পাবেন। {pct} গ্যারান্টি।",
            "আমাদের এআই ট্রেডিং বট প্রতিদিন আপনার {crypto} দ্বিগুণ করে। টেলিগ্রামে যোগ দিন: {url}",
        ],
        "lottery_prize": [
            "অভিনন্দন! আপনার নম্বর {prize} জিতেছে। পুরস্কার পেতে {fee} প্রসেসিং ফি পাঠান এবং আপনার {wallet} নম্বর দিন।",
            "আপনি আজকের ভাগ্যবান বিজয়ী! {item} জিতেছেন। ২৪ ঘণ্টার মধ্যে দাবি করুন: {url}",
        ],
        "bank_kyc": [
            "আপনার {bank} অ্যাকাউন্ট সন্দেহজনক কার্যকলাপের কারণে সাসপেন্ড করা হয়েছে। এখনই KYC আপডেট করুন: {url}",
            "আপনার এটিএম কার্ড ব্লক করা হয়েছে। ২৪ ঘণ্টার মধ্যে তথ্য আপডেট না করলে অ্যাকাউন্ট বন্ধ হবে: {url}",
        ],
        "delivery_otp": [
            "আপনার পার্সেল কাস্টমসে আটকে আছে। {fee} ছাড়ানোর ফি দিন: {url}",
            "ডেলিভারি সম্পন্ন করতে আমাদের কুরিয়ারকে আপনার OTP দিন। কোড: {otp}",
        ],
        "job": [
            "ঘরে বসে প্রতিদিন {salary} আয় করুন। কোনো অভিজ্ঞতা লাগবে না। WhatsApp করুন: {phone}",
            "জরুরি নিয়োগ: ডেটা এন্ট্রি কাজ, প্রতিদিন পেমেন্ট। টেলিগ্রামে যোগ দিন: {url}",
        ],
        "romance": [
            "প্রিয়, আমি বিদেশে সেনাবাহিনীতে আছি। তোমার সাথে দেখা করতে {amount} দরকার। {wallet}-এ পাঠাও।",
            "আমার সাথে দেখা করতে চাইলে {amount} পাঠাও, আমি টিকিট কাটব। আমি তোমাকে ভালোবাসি।",
        ],
        "marketplace": [
            "আমি আমার {item} {amount}-এ বিক্রি করছি। বিদেশে আছি, তাই শিপিংয়ের আগে {fee} ডিপোজিট পাঠাও।",
            "নতুন {item}, সিলড বক্স, মাত্র {amount}। Cash App-এ পেমেন্ট করলেই আজ শিপিং।",
        ],
        "loan_investment": [
            "{amount} লোন ০% সুদে অনুমোদিত। কোনো কাগজপত্র লাগবে না। এখনই আবেদন করুন: {url}",
            "{amount} বিনিয়োগ করুন, ৭ দিনে {pct} লাভ করুন। গ্যারান্টি! যোগাযোগ করুন: {phone}",
        ],
        "tech_support": [
            "আপনার ফোনে ভাইরাস পাওয়া গেছে। এখনই {phone}-এ কল করুন এবং রিমোট অ্যাক্সেস দিন।",
            "আপনার Apple ID লক করা হয়েছে। এখনই {phone}-এ কল করুন এবং কোডটি বলুন।",
        ],
        "promo": [
            "আজই {pct} ছাড়! কোড {code} ব্যবহার করুন: {url}",
            "আপনার পরবর্তী রিচার্জে {amount} ক্যাশব্যাক। দাবি করুন: {url}",
        ],
    },
    "Banglish": {
        "p2p_bait": [
            "Bhai, ami ekta {item} nibo. Tumi {amount} pathao, ami tomake {amount2} ferot debo.",
            "Amake {amount} pathan, ami apnake notun {item} gift dibo. Sudhu ekta baki ache.",
        ],
        "crypto": [
            "Sudhu {crypto} pathan, 10 minute e {crypto2} ferot paben. {pct} guarantee.",
            "Amader AI trading bot protidin apnar {crypto} double kore. Telegram e join korun: {url}",
        ],
        "lottery_prize": [
            "Obhinondon! Apnar number {prize} jiteche. Prize pete {fee} processing fee pathan ebong apnar {wallet} number din.",
            "Apni aajker lucky winner! {item} jitechen. 24 ghontar moddhe claim korun: {url}",
        ],
        "bank_kyc": [
            "Apnar {bank} account suspicious activity er karone suspend hoyeche. Ekhoni KYC update korun: {url}",
            "Apnar ATM card block hoyeche. 24 ghontar moddhe details update na korle account bondho hobe: {url}",
        ],
        "delivery_otp": [
            "Apnar parcel customs e atke ache. {fee} clearance fee din: {url}",
            "Delivery complete korte amader courier ke apnar OTP din. Code: {otp}",
        ],
        "job": [
            "Ghore boshe protidin {salary} income korun. Experience lagbe na. WhatsApp korun: {phone}",
            "Urgent hiring: data entry job, daily payment. Telegram e join korun: {url}",
        ],
        "romance": [
            "Priyo, ami bideshe army te achi. Tomar sathe dekha korte {amount} dorkar. {wallet} e pathao.",
            "Ami tomake bhalobashi. Amar ticket er jonno {amount} pathao, ami asbo.",
        ],
        "marketplace": [
            "Ami amar {item} {amount} taka te bechchi. Bideshe achi, tai shipping er age {fee} deposit pathao.",
            "Notun {item}, sealed box, matro {amount} taka. Cash App e payment korlei aaj ship korbo.",
        ],
        "loan_investment": [
            "{amount} loan 0% interest e approved. Kono document lagbe na. Ekhoni apply korun: {url}",
            "{amount} invest korun, 7 dine {pct} profit korun. Guarantee! Contact: {phone}",
        ],
        "tech_support": [
            "Apnar phone e virus dhora poreche. Ekhoni {phone} e call korun ebong remote access din.",
            "Apnar Apple ID lock hoyeche. Ekhoni {phone} e call korun ebong code bolun.",
        ],
        "promo": [
            "Aaj i {pct} off! Code {code} use korun: {url}",
            "Apnar next recharge e {amount} cashback. Claim korun: {url}",
        ],
    },
    "Hindi": {
        "p2p_bait": [
            "भाई, मुझे {item} चाहिए। तू {amount} भेज दे, मैं तुझे {amount2} वापस दूंगा।",
            "मुझे {amount} भेजो, मैं तुम्हें नया {item} गिफ्ट में दूंगा। सिर्फ एक बचा है।",
        ],
        "crypto": [
            "सिर्फ {crypto} भेजें और 10 मिनट में {crypto2} पाएं। {pct} गारंटीड प्रॉफिट।",
            "हमारा AI ट्रेडिंग बॉट आपका {crypto} रोज डबल करता है। टेलीग्राम जॉइन करें: {url}",
        ],
        "lottery_prize": [
            "बधाई हो! आपका नंबर {prize} का इनाम जीता है। इनाम पाने के लिए {fee} प्रोसेसिंग फीस भेजें और अपना {wallet} नंबर दें।",
            "आप आज के लकी विनर हैं! {item} जीता है। 24 घंटे में दावा करें: {url}",
        ],
        "bank_kyc": [
            "आपका {bank} अकाउंट संदिग्ध गतिविधि के कारण सस्पेंड कर दिया गया है। तुरंत KYC अपडेट करें: {url}",
            "आपका ATM कार्ड ब्लॉक हो गया है। 24 घंटे में जानकारी अपडेट नहीं की तो अकाउंट बंद हो जाएगा: {url}",
        ],
        "delivery_otp": [
            "आपका पार्सल कस्टम्स में रुका हुआ है। {fee} क्लीयरेंस फीस भेजें: {url}",
            "डिलीवरी पूरी करने के लिए हमारे कूरियर को अपना OTP दें। कोड: {otp}",
        ],
        "job": [
            "घर बैठे रोज {salary} कमाएं। कोई अनुभव नहीं चाहिए। WhatsApp करें: {phone}",
            "जरूरी भर्ती: डेटा एंट्री का काम, रोज पेमेंट। टेलीग्राम जॉइन करें: {url}",
        ],
        "romance": [
            "मेरी जान, मैं विदेश में आर्मी में हूं। तुमसे मिलने के लिए {amount} चाहिए। {wallet} पर भेज दो।",
            "मैं तुमसे प्यार करता हूं। मेरी टिकट के लिए {amount} भेजो, मैं आ जाऊंगा।",
        ],
        "marketplace": [
            "मैं अपना {item} सिर्फ {amount} में बेच रहा हूं। मैं ओवरसीज हूं, इसलिए शिपिंग से पहले {fee} डिपॉजिट भेजें।",
            "नया {item}, सील्ड बॉक्स, सिर्फ {amount}। Cash App पर पेमेंट करो, आज ही शिप करूंगा।",
        ],
        "loan_investment": [
            "{amount} का लोन 0% ब्याज पर अप्रूव्ड। कोई डॉक्यूमेंट नहीं चाहिए। अभी अप्लाई करें: {url}",
            "{amount} निवेश करें, 7 दिन में {pct} प्रॉफिट पाएं। गारंटी! संपर्क करें: {phone}",
        ],
        "tech_support": [
            "आपके फोन में वायरस मिला है। तुरंत {phone} पर कॉल करें और रिमोट एक्सेस दें।",
            "आपकी Apple ID लॉक हो गई है। तुरंत {phone} पर कॉल करें और कोड बताएं।",
        ],
        "promo": [
            "आज ही {pct} छूट! कोड {code} इस्तेमाल करें: {url}",
            "आपके अगले रिचार्ज पर {amount} कैशबैक। दावा करें: {url}",
        ],
    },
    "Hinglish": {
        "p2p_bait": [
            "Bhai, mujhe {item} chahiye. Tu {amount} bhej de, main tujhe {amount2} wapas dunga.",
            "Mujhe {amount} bhejo, main tumhe naya {item} gift mein dunga. Sirf ek bacha hai.",
        ],
        "crypto": [
            "Sirf {crypto} bhejein aur 10 minute mein {crypto2} paayein. {pct} guaranteed profit.",
            "Hamara AI trading bot aapka {crypto} roz double karta hai. Telegram join karein: {url}",
        ],
        "lottery_prize": [
            "Badhai ho! Aapka number {prize} ka inaam jeeta hai. Inaam paane ke liye {fee} processing fee bhejein aur apna {wallet} number dein.",
            "Aap aaj ke lucky winner hain! {item} jeeta hai. 24 ghante mein claim karein: {url}",
        ],
        "bank_kyc": [
            "Aapka {bank} account suspicious activity ki wajah se suspend kar diya gaya hai. Turant KYC update karein: {url}",
            "Aapka ATM card block ho gaya hai. 24 ghante mein details update nahi ki to account band ho jayega: {url}",
        ],
        "delivery_otp": [
            "Aapka parcel customs mein ruka hua hai. {fee} clearance fee bhejein: {url}",
            "Delivery complete karne ke liye hamare courier ko apna OTP dein. Code: {otp}",
        ],
        "job": [
            "Ghar baithe roz {salary} kamayein. Koi experience nahi chahiye. WhatsApp karein: {phone}",
            "Urgent hiring: data entry job, daily payment. Telegram join karein: {url}",
        ],
        "romance": [
            "Meri jaan, main videsh mein army mein hoon. Tumse milne ke liye {amount} chahiye. {wallet} par bhej do.",
            "Main tumse pyaar karta hoon. Meri ticket ke liye {amount} bhejo, main aa jaunga.",
        ],
        "marketplace": [
            "Main apna {item} sirf {amount} mein bech raha hoon. Main overseas hoon, isliye shipping se pehle {fee} deposit bhejein.",
            "Naya {item}, sealed box, sirf {amount}. Cash App par payment karo, aaj hi ship karunga.",
        ],
        "loan_investment": [
            "{amount} ka loan 0% interest par approved. Koi document nahi chahiye. Abhi apply karein: {url}",
            "{amount} invest karein, 7 din mein {pct} profit paayein. Guarantee! Contact: {phone}",
        ],
        "tech_support": [
            "Aapke phone mein virus mila hai. Turant {phone} par call karein aur remote access dein.",
            "Aapki Apple ID lock ho gayi hai. Turant {phone} par call karein aur code batayein.",
        ],
        "promo": [
            "Aaj hi {pct} off! Code {code} use karein: {url}",
            "Aapke next recharge par {amount} cashback. Claim karein: {url}",
        ],
    },
    "Spanish": {
        "p2p_bait": [
            "Amigo, necesito un favor. Envíame {amount} y te daré un {item} nuevo. Te lo juro.",
            "Solo paga {fee} de envío y te doy un {item} gratis. Es en serio, sin estafa.",
        ],
        "crypto": [
            "Envía {crypto} a esta billetera y recibe {crypto2} en 10 minutos. Ganancia garantizada del {pct}.",
            "Nuestro bot de trading duplica tu {crypto} a diario. Únete al Telegram VIP: {url}",
        ],
        "lottery_prize": [
            "¡Felicidades! Tu número ganó {prize} en la lotería de {brand}. Para reclamar, envía tu número de {wallet} y paga {fee}.",
            "Eres el ganador de hoy de un {item}. Reclama en 24 horas: {url}",
        ],
        "bank_kyc": [
            "Su cuenta de {bank} ha sido suspendida por actividad sospechosa. Verifique su KYC inmediatamente: {url}",
            "Su tarjeta fue bloqueada. Actualice sus datos en {url} o su cuenta será cerrada en 24 horas.",
        ],
        "delivery_otp": [
            "Su paquete está retenido en aduanas. Pague {fee} de liberación aquí: {url}",
            "La entrega falló. Reagende y pague {fee} en {url}. Guía {ref}.",
        ],
        "job": [
            "Trabaje desde casa y gane {salary}. No necesita experiencia. Escríbanos por WhatsApp: {phone}",
            "Contratación urgente para entrada de datos. Pago: {salary}. Únase a Telegram: {url}",
        ],
        "romance": [
            "Mi amor, soy {occupation} y estoy en el extranjero. Necesito {amount} para la visa y poder verte. Envíalo por {wallet}.",
            "Siento una conexión especial contigo. ¿Puedes ayudarme con {amount} para la operación de mi madre?",
        ],
        "marketplace": [
            "Vendo mi {item} por solo {amount}. Estoy en el extranjero, así que envíen un depósito de {fee} por {wallet} y lo envío.",
            "Tengo dos boletos para {item}. Solo Zelle. Paguen {amount} de depósito y les transfiero los boletos.",
        ],
        "loan_investment": [
            "Préstamo preaprobado de {amount} al 0% de interés. Sin documentos. Solicítelo aquí: {url}",
            "Retorno garantizado del {pct} en 7 días. Inversión mínima de {amount}. Únase ahora: {url}",
        ],
        "tech_support": [
            "Su dispositivo tiene un virus. Llame inmediatamente al {phone} y permita el acceso remoto.",
            "Alerta de Microsoft: su licencia de Windows expiró. Renuévela en {url}.",
        ],
        "promo": [
            "¡{pct} de descuento solo hoy! Use el código {code} en {url}.",
            "Oferta exclusiva: {amount} de reembolso en su próxima recarga. Reclame en {url}.",
        ],
    },
    "French": {
        "p2p_bait": [
            "Mon ami, j'ai besoin d'un service. Envoie-moi {amount} et je te donnerai un {item} neuf. Je te jure.",
            "Paie seulement {fee} de livraison et je te donne un {item} gratuit. C'est sérieux, pas d'arnaque.",
        ],
        "crypto": [
            "Envoyez {crypto} à ce portefeuille et recevez {crypto2} en 10 minutes. Profit garanti de {pct}.",
            "Notre bot de trading double votre {crypto} chaque jour. Rejoignez le Telegram VIP : {url}",
        ],
        "lottery_prize": [
            "Félicitations ! Votre numéro a gagné {prize} à la loterie {brand}. Pour réclamer, envoyez votre numéro {wallet} et payez {fee}.",
            "Vous êtes le gagnant du jour d'un {item}. Réclamez sous 24 heures : {url}",
        ],
        "bank_kyc": [
            "Votre compte {bank} est suspendu pour activité suspecte. Vérifiez votre KYC immédiatement : {url}",
            "Votre carte a été bloquée. Mettez à jour vos informations sur {url} ou votre compte sera fermé sous 24 heures.",
        ],
        "delivery_otp": [
            "Votre colis est bloqué en douane. Payez {fee} de dédouanement ici : {url}",
            "La livraison a échoué. Reprogrammez et payez {fee} sur {url}. Suivi {ref}.",
        ],
        "job": [
            "Travaillez depuis chez vous et gagnez {salary}. Aucune expérience requise. Contactez-nous sur WhatsApp : {phone}",
            "Recrutement urgent pour la saisie de données. Rémunération : {salary}. Rejoignez Telegram : {url}",
        ],
        "romance": [
            "Mon amour, je suis {occupation} à l'étranger. J'ai besoin de {amount} pour mon visa. Envoyez-le via {wallet}.",
            "Je ressens une connexion spéciale avec toi. Peux-tu m'aider avec {amount} pour l'hôpital de ma mère ?",
        ],
        "marketplace": [
            "Je vends mon {item} pour seulement {amount}. Je suis à l'étranger, envoyez un dépôt de {fee} via {wallet} et je l'expédie.",
            "J'ai deux billets pour {item}. Zelle uniquement. Payez {amount} d'acompte et je transfère les billets.",
        ],
        "loan_investment": [
            "Prêt préapprouvé de {amount} à 0 % d'intérêt. Sans documents. Demandez ici : {url}",
            "Rendement garanti de {pct} en 7 jours. Investissement minimum de {amount}. Rejoignez-nous : {url}",
        ],
        "tech_support": [
            "Votre appareil est infecté par un virus. Appelez immédiatement le {phone} et autorisez l'accès à distance.",
            "Alerte Microsoft : votre licence Windows a expiré. Renouvelez-la sur {url}.",
        ],
        "promo": [
            "{pct} de réduction aujourd'hui seulement ! Utilisez le code {code} sur {url}.",
            "Offre exclusive : {amount} de remboursement sur votre prochaine recharge. Réclamez sur {url}.",
        ],
    },
    "Arabic": {
        "p2p_bait": [
            "يا صديقي، أحتاج مساعدة. أرسل لي {amount} وسأعطيك {item} جديداً. أعدك.",
            "ادفع فقط {fee} رسوم شحن وسأعطيك {item} مجاناً. الأمر جدي، لا نصب.",
        ],
        "crypto": [
            "أرسل {crypto} إلى هذه المحفظة واحصل على {crypto2} خلال 10 دقائق. ربح مضمون {pct}.",
            "بوت التداول لدينا يضاعف {crypto} يومياً. انضم إلى تيليجرام VIP: {url}",
        ],
        "lottery_prize": [
            "مبروك! رقمك فاز بـ {prize} في يانصيب {brand}. للمطالبة، أرسل رقم {wallet} وادفع {fee}.",
            "أنت الفائز اليوم بـ {item}. طالب بالجائزة خلال 24 ساعة: {url}",
        ],
        "bank_kyc": [
            "تم تعليق حسابك في {bank} بسبب نشاط مشبوه. تحقق من KYC فوراً: {url}",
            "تم حظر بطاقتك. حدّث بياناتك على {url} وإلا سيتم إغلاق حسابك خلال 24 ساعة.",
        ],
        "delivery_otp": [
            "شحنتك محتجزة في الجمارك. ادفع {fee} رسوم تخليص هنا: {url}",
            "فشل التسليم. أعد الجدولة وادفع {fee} على {url}. رقم الشحنة {ref}.",
        ],
        "job": [
            "اعمل من المنزل واربح {salary}. لا تحتاج خبرة. تواصل معنا على واتساب: {phone}",
            "توظيف عاجل لإدخال البيانات. الراتب: {salary}. انضم إلى تيليجرام: {url}",
        ],
        "romance": [
            "حبيبي، أنا {occupation} في الخارج. أحتاج {amount} لتأشيرة السفر لأراك. أرسلها عبر {wallet}.",
            "أشعر بارتباط خاص بك. هل يمكنك مساعدتي بـ {amount} لعملية والدتي؟",
        ],
        "marketplace": [
            "أبيع {item} مقابل {amount} فقط. أنا في الخارج، أرسل {fee} عربوناً عبر {wallet} وسأشحنه.",
            "لدي تذكرتان لـ {item}. زيل فقط. ادفع {amount} عربوناً وسأرسل التذاكر.",
        ],
        "loan_investment": [
            "قرض معتمد بمبلغ {amount} بفائدة 0%. بدون مستندات. تقدم الآن: {url}",
            "عائد مضمون {pct} خلال 7 أيام. الحد الأدنى للاستثمار {amount}. انضم الآن: {url}",
        ],
        "tech_support": [
            "جهازك مصاب بفيروس. اتصل فوراً على {phone} واسمح بالوصول عن بعد.",
            "تحذير من مايكروسوفت: انتهت صلاحية ويندوز. جدد الآن على {url}.",
        ],
        "promo": [
            "خصم {pct} اليوم فقط! استخدم الرمز {code} على {url}.",
            "عرض حصري: {amount} استرداد نقدي على شحنتك القادمة. اطلب على {url}.",
        ],
    },
}

HAM_OPENERS: dict[str, list[str]] = {
    "English": ["Hi", "Hey", "Hello", "Good morning", "Thanks", "Dear", "Hi team,", "Yo"],
    "Bengali": ["হ্যালো", "প্রিয়", "শুভ সকাল", "ধন্যবাদ", "হাই"],
    "Banglish": ["Hello", "Priyo", "Shubho sokal", "Dhonnobad", "Hi"],
    "Hindi": ["नमस्ते", "प्रिय", "सुप्रभात", "धन्यवाद", "हाय"],
    "Hinglish": ["Namaste", "Priya", "Suprabhat", "Dhanyavad", "Hi"],
    "Spanish": ["Hola", "Buenos días", "Gracias", "Estimado", "Hola equipo,"],
    "French": ["Bonjour", "Salut", "Merci", "Cher", "Bonjour à tous,"],
    "Arabic": ["مرحبا", "صباح الخير", "شكرا", "عزيزي", "أهلا،"],
}

HAM_CLOSERS: dict[str, list[str]] = {
    "English": ["Let me know.", "See you soon.", "Thanks!", "Take care.", "No rush.", "Talk later."],
    "Bengali": ["জানাবেন।", "পরে কথা হবে।", "ধন্যবাদ।", "ভালো থাকবেন।"],
    "Banglish": ["Janaben.", "Pore kotha hobe.", "Dhonnobad.", "Valo thakben."],
    "Hindi": ["बताइएगा।", "बाद में बात करेंगे।", "धन्यवाद।", "अपना ख्याल रखें।"],
    "Hinglish": ["Bataiyega.", "Baad mein baat karenge.", "Dhanyavad.", "Apna khayal rakhein."],
    "Spanish": ["Avísame.", "Hasta pronto.", "Gracias.", "Cuídate."],
    "French": ["Tiens-moi au courant.", "À bientôt.", "Merci.", "Prends soin de toi."],
    "Arabic": ["أخبرني.", "أراك قريبا.", "شكرا.", "اعتن بنفسك."],
}

HAM_BODIES: dict[str, dict[str, list[str]]] = {
    "English": {
        "personal": [
            "Can you send me {amount} for lunch? I will pay you back tomorrow.",
            "Are we still meeting at {time} today? I can book the table.",
            "I reached home safely. Call me when you are free.",
            "Please pick up milk and eggs on your way back.",
            "Did you finish the assignment? I can share my notes.",
            "Happy birthday! Hope you have a great day.",
            "I left my charger at your place. Can I get it tomorrow?",
            "The movie starts at 7. Want to grab dinner first?",
        ],
        "bank_alert": [
            "Your account was debited by {amount} at {brand}. Available balance {balance}.",
            "A new login was detected on your account. If this was you, no action is needed. Visit {legit_url}.",
            "Your salary of {amount} has been credited to your account ending {acct}.",
            "Your card payment of {amount} was successful. Thank you for banking with {bank}.",
            "Your account statement for this month is ready at {legit_url}.",
        ],
        "otp": [
            "Your OTP is {otp}. Do not share it with anyone. It expires in 10 minutes.",
            "Use code {otp} to verify your login. Never share this code.",
            "Your verification code is {otp}. If you did not request it, ignore this message.",
            "Do not share your OTP or PIN with anyone, including bank staff.",
        ],
        "delivery": [
            "Your parcel is out for delivery and should arrive today. Track it at {legit_url}.",
            "Delivery attempt failed. Reschedule at {legit_url}. Your package will be held for 3 days.",
            "Your order has shipped. Track your package at {legit_url}.",
            "The courier will call you before arriving. Please keep your phone nearby.",
        ],
        "work": [
            "The meeting is moved to {time} in Room 302. Please review the deck before we start.",
            "Your salary has been credited. The payslip is available on the portal.",
            "Can you review the attached proposal and send comments by Friday?",
            "I will be working from home tomorrow. Let's sync at {time}.",
        ],
        "marketplace": [
            "I am selling my {item} for {amount}. Pickup only, no shipping. DM if interested.",
            "I can do {amount} if you pick it up today. The box is unopened.",
            "Is the {item} still available? I can meet at the station at {time}.",
            "I paid {amount} for the concert ticket. Can you transfer your share?",
        ],
        "promo_optin": [
            "Your Amazon order has shipped. Track your package at {legit_url}.",
            "Thanks for subscribing. Here is your 10% coupon code WELCOME10 for your next purchase.",
            "Your food order is confirmed and will arrive in 30 minutes. Track it in the app.",
            "Your appointment is confirmed for {time}. Reply CANCEL if you need to reschedule.",
        ],
        "urgent_legit": [
            "Mom is in the hospital. Call me as soon as you see this.",
            "Please send the signed contract by 5 PM today. The client is waiting.",
            "The deadline was moved to tomorrow. We need the final draft tonight.",
            "There is a water leak in the kitchen. Can you come home early?",
        ],
        "money_legit": [
            "I transferred {amount} to your account for the concert ticket. Check your bank.",
            "Split the bill? Your share is {amount}. I will send a request on Venmo.",
            "I sent you {amount} by mistake. Can you send it back when you get a chance?",
            "Here is the receipt from lunch. Your share is {amount}.",
        ],
        "link_legit": [
            "Here is the link to the meeting: {legit_url}",
            "Join our webinar at {legit_url}. It starts at {time}.",
            "I shared the photos here: {legit_url}",
            "The event page is live: {legit_url}. Let me know if you can make it.",
        ],
        "kyc_legit": [
            "Please complete your KYC at the bank branch by Friday. Bring your ID.",
            "Your KYC verification is complete. No further action is needed.",
            "We need a copy of your ID for the account update. You can upload it at {legit_url}.",
            "Your address verification is pending. Please visit the branch at your convenience.",
        ],
        "refund": [
            "Your refund of {amount} has been processed. It may take 3-5 business days.",
            "Your order was canceled and the amount will be refunded to the original payment method.",
            "The return was approved. Please drop the package at any {brand} store.",
            "Your refund is complete. Check your account ending {acct}.",
        ],
    },
    "Bengali": {
        "personal": [
            "দুপুরে খাবি? আমি {amount} টাকা পাঠিয়ে দিয়েছি।",
            "আজ সন্ধ্যায় দেখা হবে? আমি {time}-এ পৌঁছে যাব।",
            "বাসায় পৌঁছে ফোন দিও।",
            "আগামীকাল পরীক্ষা, নোটগুলো পাঠিয়ে দিস।",
        ],
        "bank_alert": [
            "আপনার অ্যাকাউন্ট থেকে {amount} ডেবিট হয়েছে। বর্তমান ব্যালেন্স {balance}।",
            "আপনার বেতন {amount} অ্যাকাউন্টে জমা হয়েছে।",
            "আপনার কার্ড পেমেন্ট সফল হয়েছে। ধন্যবাদ।",
            "নতুন লগইন শনাক্ত হয়েছে। এটি আপনি হলে কোনো পদক্ষেপ নিতে হবে না।",
        ],
        "otp": [
            "আপনার OTP {otp}। এটি কারও সাথে শেয়ার করবেন না।",
            "লগইন যাচাইয়ের কোড {otp}। কোডটি গোপন রাখুন।",
            "আপনার ভেরিফিকেশন কোড {otp}। আপনি অনুরোধ না করলে এটি উপেক্ষা করুন।",
        ],
        "delivery": [
            "আপনার পার্সেল আজ ডেলিভারি হবে। ট্র্যাক করুন: {legit_url}",
            "ডেলিভারি ব্যর্থ হয়েছে। পুনরায় সময় নির্ধারণ করুন: {legit_url}",
            "আপনার অর্ডার পাঠানো হয়েছে। ট্র্যাক করুন: {legit_url}",
        ],
        "work": [
            "মিটিং {time}-এ রুম ৩০২-এ অনুষ্ঠিত হবে।",
            "আপনার বেতন জমা হয়েছে। পেস্লিপ পোর্টালে আছে।",
            "প্রস্তাবনাটি দেখে শুক্রবারের মধ্যে মতামত দিন।",
        ],
        "marketplace": [
            "আমি আমার {item} {amount} টাকায় বিক্রি করছি। নিজে এসে নিতে হবে।",
            "{amount} টাকায় দিতে পারি, আজ এসে নিলে।",
            "{item} এখনো আছে? স্টেশনে দেখা করা যাবে।",
        ],
        "promo_optin": [
            "আপনার অর্ডার কনফার্ম হয়েছে। ৩০ মিনিটে পৌঁছে যাবে।",
            "সাবস্ক্রাইব করার জন্য ধন্যবাদ। আপনার কুপন কোড WELCOME10।",
            "আপনার অ্যাপয়েন্টমেন্ট {time}-এ কনফার্ম হয়েছে।",
        ],
        "urgent_legit": [
            "মা হাসপাতালে ভর্তি। এখনই ফোন দিও।",
            "চুক্তিপত্র আজ ৫টার মধ্যে পাঠাতে হবে।",
            "রান্নাঘরে পানি পড়ছে। তাড়াতাড়ি বাসায় আসো।",
        ],
        "money_legit": [
            "কনসার্টের টিকিটের জন্য {amount} টাকা পাঠিয়েছি। ব্যাংক চেক করো।",
            "বিল ভাগ করি? তোমার অংশ {amount} টাকা।",
            "ভুলে {amount} টাকা পাঠিয়ে দিয়েছি। ফেরত দিও।",
        ],
        "link_legit": [
            "মিটিংয়ের লিংক: {legit_url}",
            "ওয়েবিনারে যোগ দিন: {legit_url}",
            "ছবিগুলো এখানে শেয়ার করেছি: {legit_url}",
        ],
        "kyc_legit": [
            "শুক্রবারের মধ্যে শাখায় KYC সম্পন্ন করুন। আইডি নিয়ে আসুন।",
            "আপনার KYC যাচাই সম্পন্ন হয়েছে। আর কিছু করতে হবে না।",
            "অ্যাকাউন্ট আপডেটের জন্য আইডির কপি দরকার। {legit_url}-এ আপলোড করুন।",
        ],
        "refund": [
            "আপনার {amount} টাকা রিফান্ড প্রক্রিয়া শুরু হয়েছে। ৩-৫ কার্যদিবস লাগবে।",
            "অর্ডার বাতিল হয়েছে। টাকা মূল পেমেন্ট পদ্ধতিতে ফেরত দেওয়া হবে।",
        ],
    },
    "Banglish": {
        "personal": [
            "Dupure khabi? Ami {amount} taka pathiye diyechi.",
            "Aaj shondhaye dekha hobe? Ami {time} e pouche jabo.",
            "Bashay pouche phone dio.",
            "Kalke exam, notes gulo pathiye dis.",
        ],
        "bank_alert": [
            "Apnar account theke {amount} debit hoyeche. Current balance {balance}.",
            "Apnar salary {amount} account e joma hoyeche.",
            "Apnar card payment successful hoyeche. Dhonnobad.",
            "Notun login dhora poreche. Apni hole kono action lagbe na.",
        ],
        "otp": [
            "Apnar OTP {otp}. Eta karor sathe share korben na.",
            "Login verify er code {otp}. Code ta gopon rakhun.",
            "Apnar verification code {otp}. Apni request na korle eta ignore korun.",
        ],
        "delivery": [
            "Apnar parcel aaj delivery hobe. Track korun: {legit_url}",
            "Delivery fail hoyeche. Abar schedule korun: {legit_url}",
            "Apnar order ship hoyeche. Track korun: {legit_url}",
        ],
        "work": [
            "Meeting {time} e Room 302 te hobe.",
            "Apnar salary joma hoyeche. Payslip portal e ache.",
            "Proposal ta dekhe Friday er moddhe comment din.",
        ],
        "marketplace": [
            "Ami amar {item} {amount} takay bechchi. Nijei ashiya nite hobe.",
            "{amount} takay dite pari, aaj ashiya nile.",
            "{item} ekhono ache? Station e dekha kora jabe.",
        ],
        "promo_optin": [
            "Apnar order confirm hoyeche. 30 minute e pouche jabe.",
            "Subscribe korar jonno dhonnobad. Apnar coupon code WELCOME10.",
            "Apnar appointment {time} e confirm hoyeche.",
        ],
        "urgent_legit": [
            "Ma hospital e vorti. Ekhoni phone dio.",
            "Contract ta aaj 5 tar moddhe pathate hobe.",
            "Rannaghore pani porchhe. Taratari bashay asho.",
        ],
        "money_legit": [
            "Concert ticket er jonno {amount} taka pathiyechi. Bank check koro.",
            "Bill vag kori? Tomar part {amount} taka.",
            "Bhule {amount} taka pathiye diyechi. Ferot dio.",
        ],
        "link_legit": [
            "Meeting er link: {legit_url}",
            "Webinar e join korun: {legit_url}",
            "Chobi gulo ekhane share korechi: {legit_url}",
        ],
        "kyc_legit": [
            "Friday er moddhe branch e KYC complete korun. ID niye asun.",
            "Apnar KYC verification complete. Ar kichu lagbe na.",
            "Account update er jonno ID er copy dorkar. {legit_url} e upload korun.",
        ],
        "refund": [
            "Apnar {amount} taka refund process shuru hoyeche. 3-5 working day lagbe.",
            "Order cancel hoyeche. Taka original payment method e ferot dewa hobe.",
        ],
    },
    "Hindi": {
        "personal": [
            "Dopahar mein khana khayega? Maine {amount} bhej diye hain.",
            "Aaj shaam milna hai? Main {time} tak pahunch jaunga.",
            "Ghar pahunch kar phone karna.",
            "Kal exam hai, notes bhej dena.",
        ],
        "bank_alert": [
            "आपके खाते से {amount} डेबिट हुए हैं। उपलब्ध बैलेंस {balance}।",
            "आपका वेतन {amount} खाते में जमा हो गया है।",
            "आपका कार्ड भुगतान सफल रहा। धन्यवाद।",
            "नया लॉगिन detected हुआ है। अगर यह आप थे तो कोई कार्रवाई नहीं करनी है।",
        ],
        "otp": [
            "आपका OTP {otp} है। इसे किसी के साथ साझा न करें।",
            "लॉगिन सत्यापित करने का कोड {otp} है। कोड गुप्त रखें।",
            "आपका सत्यापन कोड {otp} है। अगर आपने अनुरोध नहीं किया तो इसे अनदेखा करें।",
        ],
        "delivery": [
            "आपका पार्सल आज डिलीवर होगा। ट्रैक करें: {legit_url}",
            "डिलीवरी असफल रही। दोबारा शेड्यूल करें: {legit_url}",
            "आपका ऑर्डर शिप हो गया है। ट्रैक करें: {legit_url}",
        ],
        "work": [
            "मीटिंग {time} बजे रूम 302 में है।",
            "आपका वेतन जमा हो गया है। पेस्लिप पोर्टल पर है।",
            "प्रस्ताव देखकर शुक्रवार तक अपनी टिप्पणी भेजें।",
        ],
        "marketplace": [
            "मैं अपना {item} {amount} में बेच रहा हूं। खुद आकर लेना होगा।",
            "{amount} में दे सकता हूं, आज आकर ले लो।",
            "{item} अभी भी उपलब्ध है? स्टेशन पर मिल सकते हैं।",
        ],
        "promo_optin": [
            "आपका ऑर्डर कन्फर्म हो गया है। 30 मिनट में पहुंच जाएगा।",
            "सब्सक्राइब करने के लिए धन्यवाद। आपका कूपन कोड WELCOME10 है।",
            "आपका अपॉइंटमेंट {time} पर कन्फर्म है।",
        ],
        "urgent_legit": [
            "मां अस्पताल में भर्ती हैं। अभी फोन करो।",
            "अनुबंध आज 5 बजे तक भेजना है।",
            "रसोई में पानी लीक हो रहा है। जल्दी घर आओ।",
        ],
        "money_legit": [
            "कॉन्सर्ट टिकट के लिए {amount} भेज दिए हैं। बैंक चेक करो।",
            "बिल बांट लें? तुम्हारा हिस्सा {amount} है।",
            "गलती से {amount} भेज दिए हैं। लौटा देना।",
        ],
        "link_legit": [
            "मीटिंग का लिंक: {legit_url}",
            "वेबिनार में शामिल हों: {legit_url}",
            "फोटो यहां शेयर की हैं: {legit_url}",
        ],
        "kyc_legit": [
            "शुक्रवार तक बैंक शाखा में KYC पूरा करें। आईडी लेकर आएं।",
            "आपका KYC सत्यापन पूरा हो गया है। अब कुछ नहीं करना है।",
            "खाता अपडेट के लिए आईडी की कॉपी चाहिए। {legit_url} पर अपलोड करें।",
        ],
        "refund": [
            "आपका {amount} रिफंड प्रोसेस हो गया है। 3-5 कार्यदिवस लग सकते हैं।",
            "आपका ऑर्डर रद्द हो गया है। राशि मूल भुगतान विधि में वापस आएगी।",
        ],
    },
    "Hinglish": {
        "personal": [
            "Dopahar mein khana khayega? Maine {amount} bhej diye hain.",
            "Aaj shaam milna hai? Main {time} tak pahunch jaunga.",
            "Ghar pahunch kar phone karna.",
            "Kal exam hai, notes bhej dena.",
        ],
        "bank_alert": [
            "Aapke account se {amount} debit hue hain. Available balance {balance}.",
            "Aapka salary {amount} account mein jama ho gaya hai.",
            "Aapka card payment successful raha. Dhanyavad.",
            "Naya login detect hua hai. Agar yeh aap the to koi action nahi karna.",
        ],
        "otp": [
            "Aapka OTP {otp} hai. Ise kisi ke saath share na karein.",
            "Login verify karne ka code {otp} hai. Code gopon rakhein.",
            "Aapka verification code {otp} hai. Aapne request nahi ki to ignore karein.",
        ],
        "delivery": [
            "Aapka parcel aaj deliver hoga. Track karein: {legit_url}",
            "Delivery fail ho gayi. Dobara schedule karein: {legit_url}",
            "Aapka order ship ho gaya hai. Track karein: {legit_url}",
        ],
        "work": [
            "Meeting {time} baje Room 302 mein hai.",
            "Aapka salary jama ho gaya hai. Payslip portal par hai.",
            "Proposal dekhkar Friday tak comment bhejein.",
        ],
        "marketplace": [
            "Main apna {item} {amount} mein bech raha hoon. Khud aakar lena hoga.",
            "{amount} mein de sakta hoon, aaj aakar le lo.",
            "{item} abhi bhi available hai? Station par mil sakte hain.",
        ],
        "promo_optin": [
            "Aapka order confirm ho gaya hai. 30 minute mein pahunch jayega.",
            "Subscribe karne ke liye dhanyavad. Aapka coupon code WELCOME10 hai.",
            "Aapka appointment {time} par confirm hai.",
        ],
        "urgent_legit": [
            "Maa hospital mein bharti hain. Abhi phone karo.",
            "Contract aaj 5 baje tak bhejna hai.",
            "Kitchen mein paani leak ho raha hai. Jaldi ghar aao.",
        ],
        "money_legit": [
            "Concert ticket ke liye {amount} bhej diye hain. Bank check karo.",
            "Bill baant lein? Tumhara hissa {amount} hai.",
            "Galti se {amount} bhej diye hain. Lauta dena.",
        ],
        "link_legit": [
            "Meeting ka link: {legit_url}",
            "Webinar mein shamil hon: {legit_url}",
            "Photos yahan share ki hain: {legit_url}",
        ],
        "kyc_legit": [
            "Friday tak bank branch mein KYC poora karein. ID lekar aayein.",
            "Aapka KYC verification poora ho gaya hai. Ab kuch nahi karna hai.",
            "Account update ke liye ID ki copy chahiye. {legit_url} par upload karein.",
        ],
        "refund": [
            "Aapka {amount} refund process ho gaya hai. 3-5 working days lag sakte hain.",
            "Aapka order cancel ho gaya hai. Raqam original payment method mein wapas aayegi.",
        ],
    },
    "Spanish": {
        "personal": [
            "¿Me puedes enviar {amount} para el almuerzo? Te lo devuelvo mañana.",
            "¿Seguimos quedando a las {time}? Puedo reservar la mesa.",
            "Ya llegué a casa. Llámame cuando puedas.",
            "Compra leche y huevos de camino a casa.",
        ],
        "bank_alert": [
            "Su cuenta fue debitada por {amount} en {brand}. Saldo disponible {balance}.",
            "Se detectó un nuevo inicio de sesión. Si fue usted, no necesita hacer nada. Visite {legit_url}.",
            "Su salario de {amount} fue abonado a su cuenta terminada en {acct}.",
            "Su pago con tarjeta de {amount} fue exitoso. Gracias por su compra.",
        ],
        "otp": [
            "Su código OTP es {otp}. No lo comparta con nadie. Expira en 10 minutos.",
            "Use el código {otp} para verificar su inicio de sesión. Nunca comparta este código.",
            "Su código de verificación es {otp}. Si no lo solicitó, ignore este mensaje.",
        ],
        "delivery": [
            "Su paquete está en camino y llegará hoy. Rastréelo en {legit_url}.",
            "Falló la entrega. Reagende en {legit_url}. El paquete se guardará 3 días.",
            "Su pedido ha sido enviado. Rastree su paquete en {legit_url}.",
        ],
        "work": [
            "La reunión se movió a las {time} en la sala 302. Revise la presentación antes.",
            "Su salario fue abonado. La nómina está disponible en el portal.",
            "¿Puede revisar la propuesta y enviar comentarios antes del viernes?",
        ],
        "marketplace": [
            "Vendo mi {item} por {amount}. Solo recogida, sin envío. Escríbeme si te interesa.",
            "Puedo dejarlo en {amount} si lo recoges hoy. La caja está sin abrir.",
            "¿El {item} sigue disponible? Podemos vernos en la estación.",
        ],
        "promo_optin": [
            "Su pedido de Amazon ha sido enviado. Rastréelo en {legit_url}.",
            "Gracias por suscribirte. Aquí tienes tu cupón del 10%: WELCOME10.",
            "Su pedido de comida está confirmado y llegará en 30 minutos.",
        ],
        "urgent_legit": [
            "Mamá está en el hospital. Llámame en cuanto veas esto.",
            "Por favor envía el contrato firmado antes de las 5 PM. El cliente espera.",
            "Hay una fuga de agua en la cocina. ¿Puedes venir temprano?",
        ],
        "money_legit": [
            "Te transferí {amount} por el boleto del concierto. Revisa tu banco.",
            "¿Dividimos la cuenta? Tu parte es {amount}. Te envío una solicitud por Venmo.",
            "Te envié {amount} por error. ¿Puedes devolverlo cuando puedas?",
        ],
        "link_legit": [
            "Aquí está el enlace de la reunión: {legit_url}",
            "Únete a nuestro webinar en {legit_url}. Empieza a las {time}.",
            "Compartí las fotos aquí: {legit_url}",
        ],
        "kyc_legit": [
            "Complete su KYC en la sucursal del banco antes del viernes. Traiga su identificación.",
            "Su verificación KYC está completa. No se requiere ninguna acción.",
            "Necesitamos una copia de su identificación. Puede subirla en {legit_url}.",
        ],
        "refund": [
            "Su reembolso de {amount} ha sido procesado. Puede tardar de 3 a 5 días hábiles.",
            "Su pedido fue cancelado y el monto se reembolsará al método de pago original.",
        ],
    },
    "French": {
        "personal": [
            "Tu peux m'envoyer {amount} pour le déjeuner ? Je te rembourse demain.",
            "On se voit toujours à {time} ? Je peux réserver la table.",
            "Je suis bien rentré. Appelle-moi quand tu es libre.",
            "Peux-tu acheter du lait et des œufs en rentrant ?",
        ],
        "bank_alert": [
            "Votre compte a été débité de {amount} chez {brand}. Solde disponible {balance}.",
            "Une nouvelle connexion a été détectée. Si c'était vous, aucune action n'est nécessaire. Visitez {legit_url}.",
            "Votre salaire de {amount} a été crédité sur votre compte se terminant par {acct}.",
            "Votre paiement par carte de {amount} a réussi. Merci de votre confiance.",
        ],
        "otp": [
            "Votre OTP est {otp}. Ne le partagez avec personne. Il expire dans 10 minutes.",
            "Utilisez le code {otp} pour vérifier votre connexion. Ne partagez jamais ce code.",
            "Votre code de vérification est {otp}. Si vous n'êtes pas à l'origine de cette demande, ignorez ce message.",
        ],
        "delivery": [
            "Votre colis est en cours de livraison et arrivera aujourd'hui. Suivez-le sur {legit_url}.",
            "La livraison a échoué. Reprogrammez sur {legit_url}. Le colis sera conservé 3 jours.",
            "Votre commande a été expédiée. Suivez votre colis sur {legit_url}.",
        ],
        "work": [
            "La réunion est déplacée à {time} en salle 302. Merci de relire la présentation.",
            "Votre salaire a été versé. La fiche de paie est disponible sur le portail.",
            "Peux-tu relire la proposition et envoyer tes commentaires avant vendredi ?",
        ],
        "marketplace": [
            "Je vends mon {item} pour {amount}. Retrait uniquement, pas d'envoi. Écris-moi si intéressé.",
            "Je peux le laisser à {amount} si tu viens le chercher aujourd'hui. La boîte est neuve.",
            "Le {item} est toujours disponible ? On peut se retrouver à la gare.",
        ],
        "promo_optin": [
            "Votre commande Amazon a été expédiée. Suivez-la sur {legit_url}.",
            "Merci pour ton abonnement. Voici ton coupon de 10 % : WELCOME10.",
            "Votre commande de nourriture est confirmée et arrivera dans 30 minutes.",
        ],
        "urgent_legit": [
            "Maman est à l'hôpital. Appelle-moi dès que tu vois ce message.",
            "Merci d'envoyer le contrat signé avant 17 h. Le client attend.",
            "Il y a une fuite d'eau dans la cuisine. Tu peux rentrer plus tôt ?",
        ],
        "money_legit": [
            "Je t'ai transféré {amount} pour le billet de concert. Vérifie ton compte.",
            "On partage l'addition ? Ta part est de {amount}. Je t'envoie une demande sur Venmo.",
            "Je t'ai envoyé {amount} par erreur. Tu peux me le renvoyer quand tu peux ?",
        ],
        "link_legit": [
            "Voici le lien de la réunion : {legit_url}",
            "Rejoins notre webinaire sur {legit_url}. Ça commence à {time}.",
            "J'ai partagé les photos ici : {legit_url}",
        ],
        "kyc_legit": [
            "Merci de compléter votre KYC en agence avant vendredi. Apportez votre pièce d'identité.",
            "Votre vérification KYC est terminée. Aucune action supplémentaire n'est nécessaire.",
            "Nous avons besoin d'une copie de votre pièce d'identité. Vous pouvez la téléverser sur {legit_url}.",
        ],
        "refund": [
            "Votre remboursement de {amount} a été traité. Il peut prendre 3 à 5 jours ouvrés.",
            "Votre commande a été annulée et le montant sera remboursé sur le moyen de paiement d'origine.",
        ],
    },
    "Arabic": {
        "personal": [
            "هل يمكنك إرسال {amount} للغداء؟ سأردها لك غداً.",
            "هل ما زلنا نلتقي في {time}؟ يمكنني حجز الطاولة.",
            "وصلت إلى المنزل بسلام. اتصل بي عندما تكون متفرغاً.",
            "اشتر الحليب والبيض في طريقك إلى المنزل.",
        ],
        "bank_alert": [
            "تم خصم {amount} من حسابك لدى {brand}. الرصيد المتاح {balance}.",
            "تم اكتشاف تسجيل دخول جديد. إذا كنت أنت، لا حاجة لأي إجراء. تفضل بزيارة {legit_url}.",
            "تم إيداع راتبك البالغ {amount} في حسابك المنتهي بـ {acct}.",
            "تمت عملية الدفع بالبطاقة بمبلغ {amount} بنجاح. شكراً لك.",
        ],
        "otp": [
            "رمز OTP الخاص بك هو {otp}. لا تشاركه مع أي شخص. تنتهي صلاحيته خلال 10 دقائق.",
            "استخدم الرمز {otp} للتحقق من تسجيل الدخول. لا تشارك هذا الرمز أبداً.",
            "رمز التحقق الخاص بك هو {otp}. إذا لم تطلبه، تجاهل هذه الرسالة.",
        ],
        "delivery": [
            "شحنتك في الطريق وستصل اليوم. تتبعها على {legit_url}.",
            "فشل التسليم. أعد الجدولة على {legit_url}. سيتم الاحتفاظ بالشحنة 3 أيام.",
            "تم شحن طلبك. تتبع شحنتك على {legit_url}.",
        ],
        "work": [
            "تم نقل الاجتماع إلى {time} في القاعة 302. يرجى مراجعة العرض قبل الحضور.",
            "تم إيداع راتبك. كشف الراتب متاح على البوابة.",
            "هل يمكنك مراجعة العرض وإرسال ملاحظاتك قبل الجمعة؟",
        ],
        "marketplace": [
            "أبيع {item} مقابل {amount}. الاستلام فقط، بدون شحن. راسلني إذا كنت مهتماً.",
            "يمكنني تركه بـ {amount} إذا استلمته اليوم. الصندوق جديد.",
            "هل {item} لا يزال متاحاً؟ يمكننا الالتقاء في المحطة.",
        ],
        "promo_optin": [
            "تم شحن طلب أمازون الخاص بك. تتبعه على {legit_url}.",
            "شكراً لاشتراكك. إليك قسيمة خصم 10%: WELCOME10.",
            "تم تأكيد طلب الطعام وسيصل خلال 30 دقيقة.",
        ],
        "urgent_legit": [
            "أمي في المستشفى. اتصل بي فور رؤية هذه الرسالة.",
            "يرجى إرسال العقد الموقع قبل الساعة 5 مساءً. العميل ينتظر.",
            "هناك تسرب مياه في المطبخ. هل يمكنك العودة مبكراً؟",
        ],
        "money_legit": [
            "أرسلت لك {amount} مقابل تذكرة الحفل. تحقق من حسابك.",
            "هل نقسم الفاتورة؟ نصيبك {amount}. سأرسل لك طلباً عبر Venmo.",
            "أرسلت لك {amount} بالخطأ. هل يمكنك إرجاعه عندما تستطيع؟",
        ],
        "link_legit": [
            "إليك رابط الاجتماع: {legit_url}",
            "انضم إلى الندوة على {legit_url}. تبدأ الساعة {time}.",
            "شاركت الصور هنا: {legit_url}",
        ],
        "kyc_legit": [
            "يرجى إكمال KYC في فرع البنك قبل الجمعة. أحضر هويتك.",
            "تم التحقق من KYC الخاص بك. لا حاجة لأي إجراء آخر.",
            "نحتاج نسخة من هويتك. يمكنك رفعها على {legit_url}.",
        ],
        "refund": [
            "تمت معالجة استرداد {amount}. قد يستغرق 3-5 أيام عمل.",
            "تم إلغاء طلبك وسيتم استرداد المبلغ إلى وسيلة الدفع الأصلية.",
        ],
    },
}

SLOTS: dict[str, dict[str, list[str]]] = {
    "English": {
        "amount": ["$5", "$20", "$50", "$100", "$250", "$500"],
        "amount2": ["$500", "$1,000", "$2,500", "$5,000"],
        "prize": ["$1,000", "$10,000", "$50,000", "$500,000"],
        "fee": ["$2.99", "$5", "$9.99", "$25"],
        "salary": ["$300/day", "$500 per day", "$2,000/week"],
        "balance": ["$1,250.00", "$840.50", "$12,300.00"],
        "crypto": ["0.1 BTC", "0.05 ETH", "100 USDT", "500 USDT"],
        "crypto2": ["0.5 BTC", "0.25 ETH", "1,000 USDT", "5,000 USDT"],
        "pct": ["100%", "200%", "300%", "500%"],
        "brand": ["Amazon", "PayPal", "Netflix", "Apple", "Microsoft", "Walmart", "DHL"],
        "bank": ["Chase", "Bank of America", "Wells Fargo", "HSBC", "Citibank", "SBI", "ICICI", "HDFC"],
        "wallet": ["PayPal", "Cash App", "Venmo", "Zelle", "bKash", "Nagad", "UPI"],
        "item": ["RTX 3090", "PS5", "iPhone 16 Pro", "MacBook Pro", "Rolex watch", "concert tickets", "gaming laptop"],
        "occupation": ["soldier", "doctor", "engineer", "widow"],
        "time": ["10 AM", "2 PM", "6 PM", "7:30 PM"],
        "ref": ["AWB 7842-1190", "TRK-55219", "PKG 90341"],
        "code": ["SAVE20", "WIN50", "GIFT10", "VIP100"],
        "otp": ["482913", "739201", "105842", "662911"],
        "acct": ["4821", "9032", "4421"],
        "url": [
            "http://secure-verify.xyz", "https://bit.ly/3xYzAbC",
            "http://track-parcel.info", "https://amaz0n-support.top",
            "http://bkash-verify.xyz", "https://paypa1-secure.click",
            "http://tinyurl.com/kyc-update", "https://bank-alert.live",
        ],
        "legit_url": [
            "https://www.amazon.com/orders", "https://www.paypal.com/signin",
            "https://track.dhl.com/ABC123", "https://meet.google.com/abc-defg-hij",
            "https://www.icicibank.com", "https://www.chase.com/security",
            "https://zoom.us/j/123456789", "https://bit.ly/team-photo",
        ],
        "email": ["support@secure-verify.xyz", "help@amaz0n-support.top"],
    },
    "Bengali": {
        "amount": ["৫০০", "১,০০০", "৫,০০০", "১০,০০০"],
        "amount2": ["৫০,০০০", "১,০০,০০০", "৫,০০,০০০"],
        "prize": ["৫০,০০০", "৫,০০,০০০", "১০,০০,০০০"],
        "fee": ["২০০", "৫০০", "১,০০০"],
        "salary": ["১,০০০", "২,০০০", "১০,০০০"],
        "balance": ["১২,৫০০", "৮,৪০০", "১,২৩,০০০"],
        "crypto": ["০.১ বিটিসি", "০.০৫ ইথেরিয়াম", "১০০ ইউএসডিটি"],
        "crypto2": ["০.৫ বিটিসি", "০.২৫ ইথেরিয়াম", "১,০০০ ইউএসডিটি"],
        "pct": ["১০০%", "২০০%", "৩০০%", "৫০০%"],
        "brand": ["অ্যামাজন", "পেপাল", "নেটফ্লিক্স", "বিকাশ", "নগদ"],
        "bank": ["সোনালী ব্যাংক", "ব্র্যাক ব্যাংক", "ডাচ-বাংলা ব্যাংক", "বিকাশ", "নগদ"],
        "wallet": ["বিকাশ", "নগদ", "রকেট", "পেপাল"],
        "item": ["আরটিএক্স ৩০৯০", "পিএস৫", "আইফোন ১৬ প্রো", "ম্যাকবুক প্রো", "কনসার্ট টিকিট"],
        "occupation": ["সেনা সদস্য", "ডাক্তার", "ইঞ্জিনিয়ার", "বিধবা"],
        "time": ["সকাল ১০টা", "দুপুর ২টা", "সন্ধ্যা ৬টা", "রাত ৮টা"],
        "ref": ["এডব্লিউবি ৭৮৪২", "টিআরকে ৫৫২১৯", "প্যাকেজ ৯০৩৪১"],
        "code": ["SAVE20", "WIN50", "GIFT10", "VIP100"],
        "otp": ["৪৮২৯১৩", "৭৩৯২০১", "১০৫৮৪২", "৬৬২৯১১"],
        "acct": ["৪৮২১", "৯০৩২", "৪৪২১"],
        "url": [
            "http://secure-verify.xyz", "https://bit.ly/3xYzAbC",
            "http://track-parcel.info", "http://bkash-verify.xyz",
            "https://paypa1-secure.click", "https://bank-alert.live",
        ],
        "legit_url": [
            "https://www.bkash.com", "https://www.grameenphone.com",
            "https://www.daraz.com.bd", "https://meet.google.com/abc-defg-hij",
            "https://www.bracbank.com", "https://bit.ly/office-map",
        ],
        "email": ["support@secure-verify.xyz", "help@bkash-verify.xyz"],
    },
    "Banglish": {
        "amount": ["500", "1,000", "5,000", "10,000"],
        "amount2": ["50,000", "1,00,000", "5,00,000"],
        "prize": ["50,000", "5,00,000", "10,00,000"],
        "fee": ["200", "500", "1,000"],
        "salary": ["1,000", "2,000", "10,000"],
        "balance": ["12,500", "8,400", "1,23,000"],
        "crypto": ["0.1 BTC", "0.05 ETH", "100 USDT"],
        "crypto2": ["0.5 BTC", "0.25 ETH", "1,000 USDT"],
        "pct": ["100%", "200%", "300%", "500%"],
        "brand": ["Amazon", "PayPal", "Netflix", "bKash", "Nagad"],
        "bank": ["Sonali Bank", "BRAC Bank", "Dutch-Bangla Bank", "bKash", "Nagad"],
        "wallet": ["bKash", "Nagad", "Rocket", "PayPal"],
        "item": ["RTX 3090", "PS5", "iPhone 16 Pro", "MacBook Pro", "concert ticket"],
        "occupation": ["soldier", "doctor", "engineer", "widow"],
        "time": ["sokal 10ta", "dupur 2ta", "shondha 6ta", "rat 8ta"],
        "ref": ["AWB 7842", "TRK 55219", "PKG 90341"],
        "code": ["SAVE20", "WIN50", "GIFT10", "VIP100"],
        "otp": ["482913", "739201", "105842", "662911"],
        "acct": ["4821", "9032", "4421"],
        "url": [
            "http://secure-verify.xyz", "https://bit.ly/3xYzAbC",
            "http://track-parcel.info", "http://bkash-verify.xyz",
            "https://paypa1-secure.click", "https://bank-alert.live",
        ],
        "legit_url": [
            "https://www.bkash.com", "https://www.grameenphone.com",
            "https://www.daraz.com.bd", "https://meet.google.com/abc-defg-hij",
            "https://www.bracbank.com", "https://bit.ly/office-map",
        ],
        "email": ["support@secure-verify.xyz", "help@bkash-verify.xyz"],
    },
    "Hindi": {
        "amount": ["₹500", "₹1,000", "₹5,000", "₹10,000"],
        "amount2": ["₹50,000", "₹1,00,000", "₹5,00,000"],
        "prize": ["₹50,000", "₹5,00,000", "₹10,00,000"],
        "fee": ["₹200", "₹500", "₹1,000"],
        "salary": ["₹1,000", "₹2,000", "₹10,000"],
        "balance": ["₹12,500", "₹8,400", "₹1,23,000"],
        "crypto": ["0.1 BTC", "0.05 ETH", "100 USDT"],
        "crypto2": ["0.5 BTC", "0.25 ETH", "1,000 USDT"],
        "pct": ["100%", "200%", "300%", "500%"],
        "brand": ["Amazon", "PayPal", "Netflix", "Paytm", "PhonePe"],
        "bank": ["SBI", "ICICI", "HDFC", "Axis Bank", "Kotak"],
        "wallet": ["Paytm", "PhonePe", "Google Pay", "UPI", "PayPal"],
        "item": ["RTX 3090", "PS5", "iPhone 16 Pro", "MacBook Pro", "कॉन्सर्ट टिकट"],
        "occupation": ["सैनिक", "डॉक्टर", "इंजीनियर", "विधवा"],
        "time": ["सुबह 10 बजे", "दोपहर 2 बजे", "शाम 6 बजे", "रात 8 बजे"],
        "ref": ["AWB 7842", "TRK 55219", "PKG 90341"],
        "code": ["SAVE20", "WIN50", "GIFT10", "VIP100"],
        "otp": ["482913", "739201", "105842", "662911"],
        "acct": ["4821", "9032", "4421"],
        "url": [
            "http://secure-verify.xyz", "https://bit.ly/3xYzAbC",
            "http://track-parcel.info", "http://sbi-kyc.xyz",
            "https://paypa1-secure.click", "https://bank-alert.live",
        ],
        "legit_url": [
            "https://www.icicibank.com", "https://www.hdfcbank.com",
            "https://www.amazon.in/orders", "https://meet.google.com/abc-defg-hij",
            "https://www.paytm.com", "https://bit.ly/office-map",
        ],
        "email": ["support@secure-verify.xyz", "help@sbi-kyc.xyz"],
    },
    "Hinglish": {
        "amount": ["₹500", "₹1,000", "₹5,000", "₹10,000"],
        "amount2": ["₹50,000", "₹1,00,000", "₹5,00,000"],
        "prize": ["₹50,000", "₹5,00,000", "₹10,00,000"],
        "fee": ["₹200", "₹500", "₹1,000"],
        "salary": ["₹1,000", "₹2,000", "₹10,000"],
        "balance": ["₹12,500", "₹8,400", "₹1,23,000"],
        "crypto": ["0.1 BTC", "0.05 ETH", "100 USDT"],
        "crypto2": ["0.5 BTC", "0.25 ETH", "1,000 USDT"],
        "pct": ["100%", "200%", "300%", "500%"],
        "brand": ["Amazon", "PayPal", "Netflix", "Paytm", "PhonePe"],
        "bank": ["SBI", "ICICI", "HDFC", "Axis Bank", "Kotak"],
        "wallet": ["Paytm", "PhonePe", "Google Pay", "UPI", "PayPal"],
        "item": ["RTX 3090", "PS5", "iPhone 16 Pro", "MacBook Pro", "concert ticket"],
        "occupation": ["soldier", "doctor", "engineer", "widow"],
        "time": ["subah 10 baje", "dopahar 2 baje", "shaam 6 baje", "raat 8 baje"],
        "ref": ["AWB 7842", "TRK 55219", "PKG 90341"],
        "code": ["SAVE20", "WIN50", "GIFT10", "VIP100"],
        "otp": ["482913", "739201", "105842", "662911"],
        "acct": ["4821", "9032", "4421"],
        "url": [
            "http://secure-verify.xyz", "https://bit.ly/3xYzAbC",
            "http://track-parcel.info", "http://sbi-kyc.xyz",
            "https://paypa1-secure.click", "https://bank-alert.live",
        ],
        "legit_url": [
            "https://www.icicibank.com", "https://www.hdfcbank.com",
            "https://www.amazon.in/orders", "https://meet.google.com/abc-defg-hij",
            "https://www.paytm.com", "https://bit.ly/office-map",
        ],
        "email": ["support@secure-verify.xyz", "help@sbi-kyc.xyz"],
    },
    "Spanish": {
        "amount": ["$500", "$1,000", "$5,000", "$10,000"],
        "amount2": ["$50,000", "$100,000", "$500,000"],
        "prize": ["$50,000", "$500,000", "$1,000,000"],
        "fee": ["$50", "$100", "$500"],
        "salary": ["$500 al día", "$1,000 por día", "$10,000 por semana"],
        "balance": ["$12,500", "$8,400", "$123,000"],
        "crypto": ["0.1 BTC", "0.05 ETH", "100 USDT"],
        "crypto2": ["0.5 BTC", "0.25 ETH", "1,000 USDT"],
        "pct": ["100%", "200%", "300%", "500%"],
        "brand": ["Amazon", "PayPal", "Netflix", "BBVA", "Santander"],
        "bank": ["BBVA", "Santander", "Banorte", "HSBC", "Citibanamex"],
        "wallet": ["PayPal", "Cash App", "Zelle", "BBVA", "Mercado Pago"],
        "item": ["RTX 3090", "PS5", "iPhone 16 Pro", "MacBook Pro", "boletos de concierto"],
        "occupation": ["soldado", "médico", "ingeniero", "viuda"],
        "time": ["10 AM", "2 PM", "6 PM", "8 PM"],
        "ref": ["AWB 7842", "TRK 55219", "PKG 90341"],
        "code": ["SAVE20", "WIN50", "GIFT10", "VIP100"],
        "otp": ["482913", "739201", "105842", "662911"],
        "acct": ["4821", "9032", "4421"],
        "url": [
            "http://secure-verify.xyz", "https://bit.ly/3xYzAbC",
            "http://track-parcel.info", "http://bbva-seguro.xyz",
            "https://paypa1-secure.click", "https://bank-alert.live",
        ],
        "legit_url": [
            "https://www.bbva.mx", "https://www.santander.com.mx",
            "https://www.amazon.com.mx/orders", "https://meet.google.com/abc-defg-hij",
            "https://www.mercadopago.com.mx", "https://bit.ly/fotos-evento",
        ],
        "email": ["soporte@secure-verify.xyz", "ayuda@bbva-seguro.xyz"],
    },
    "French": {
        "amount": ["50 €", "100 €", "500 €", "1 000 €"],
        "amount2": ["5 000 €", "10 000 €", "50 000 €"],
        "prize": ["10 000 €", "100 000 €", "500 000 €"],
        "fee": ["5 €", "10 €", "50 €"],
        "salary": ["300 €/jour", "500 € par jour", "2 000 €/semaine"],
        "balance": ["1 250 €", "840 €", "12 300 €"],
        "crypto": ["0.1 BTC", "0.05 ETH", "100 USDT"],
        "crypto2": ["0.5 BTC", "0.25 ETH", "1 000 USDT"],
        "pct": ["100 %", "200 %", "300 %", "500 %"],
        "brand": ["Amazon", "PayPal", "Netflix", "BNP Paribas", "Société Générale"],
        "bank": ["BNP Paribas", "Société Générale", "Crédit Agricole", "HSBC", "La Banque Postale"],
        "wallet": ["PayPal", "Cash App", "Zelle", "Lydia", "Revolut"],
        "item": ["RTX 3090", "PS5", "iPhone 16 Pro", "MacBook Pro", "billets de concert"],
        "occupation": ["soldat", "médecin", "ingénieur", "veuve"],
        "time": ["10 h", "14 h", "18 h", "20 h"],
        "ref": ["AWB 7842", "TRK 55219", "PKG 90341"],
        "code": ["SAVE20", "WIN50", "GIFT10", "VIP100"],
        "otp": ["482913", "739201", "105842", "662911"],
        "acct": ["4821", "9032", "4421"],
        "url": [
            "http://secure-verify.xyz", "https://bit.ly/3xYzAbC",
            "http://track-parcel.info", "http://paypal-verification.top",
            "https://paypa1-secure.click", "https://bank-alert.live",
        ],
        "legit_url": [
            "https://www.bnpparibas.fr", "https://www.societegenerale.fr",
            "https://www.amazon.fr/orders", "https://meet.google.com/abc-defg-hij",
            "https://www.paypal.com/fr/signin", "https://bit.ly/photos-event",
        ],
        "email": ["support@secure-verify.xyz", "aide@paypal-verification.top"],
    },
    "Arabic": {
        "amount": ["500 ريال", "1,000 ريال", "5,000 ريال", "10,000 ريال"],
        "amount2": ["50,000 ريال", "100,000 ريال", "500,000 ريال"],
        "prize": ["50,000 ريال", "500,000 ريال", "1,000,000 ريال"],
        "fee": ["50 ريال", "100 ريال", "500 ريال"],
        "salary": ["500 ريال يومياً", "1,000 ريال يومياً", "10,000 ريال أسبوعياً"],
        "balance": ["12,500 ريال", "8,400 ريال", "123,000 ريال"],
        "crypto": ["0.1 BTC", "0.05 ETH", "100 USDT"],
        "crypto2": ["0.5 BTC", "0.25 ETH", "1,000 USDT"],
        "pct": ["100%", "200%", "300%", "500%"],
        "brand": ["أمازون", "باي بال", "نتفليكس", "الراجحي", "الأهلي"],
        "bank": ["الراجحي", "الأهلي", "الإنماء", "HSBC", "بنك الرياض"],
        "wallet": ["باي بال", "Cash App", "زيل", "STC Pay", "مدى"],
        "item": ["RTX 3090", "PS5", "iPhone 16 Pro", "MacBook Pro", "تذاكر حفل"],
        "occupation": ["جندي", "طبيب", "مهندس", "أرملة"],
        "time": ["10 صباحاً", "2 مساءً", "6 مساءً", "8 مساءً"],
        "ref": ["AWB 7842", "TRK 55219", "PKG 90341"],
        "code": ["SAVE20", "WIN50", "GIFT10", "VIP100"],
        "otp": ["482913", "739201", "105842", "662911"],
        "acct": ["4821", "9032", "4421"],
        "url": [
            "http://secure-verify.xyz", "https://bit.ly/3xYzAbC",
            "http://track-parcel.info", "http://bank-verify.xyz",
            "https://paypa1-secure.click", "https://bank-alert.live",
        ],
        "legit_url": [
            "https://www.alrajhibank.com.sa", "https://www.alahli.com",
            "https://www.amazon.sa/orders", "https://meet.google.com/abc-defg-hij",
            "https://www.stcpay.com.sa", "https://bit.ly/office-map",
        ],
        "email": ["support@secure-verify.xyz", "help@bank-verify.xyz"],
    },
}

DEFAULT_PHONES = {
    "English": ["+1 202-555-0199", "+44 7700 900123", "+8801712345678", "+91 98765 43210"],
    "Bengali": ["+8801712345678", "01712345678", "+8801912345678"],
    "Banglish": ["+8801712345678", "01712345678", "+8801912345678"],
    "Hindi": ["+91 98765 43210", "1800-123-4567", "+91 91234 56789"],
    "Hinglish": ["+91 98765 43210", "1800-123-4567", "+91 91234 56789"],
    "Spanish": ["+52 55 1234 5678", "800-555-0199", "+34 612 345 678"],
    "French": ["+33 6 12 34 56 78", "01 23 45 67 89"],
    "Arabic": ["+966 50 123 4567", "+971 50 123 4567", "+20 10 1234 5678"],
}

for _lang, _pool in SLOTS.items():
    _pool.setdefault("phone", DEFAULT_PHONES.get(_lang, ["+1 202-555-0199"]))

SCAM_URL_OBFUSCATIONS = ("bracket", "paren", "dotword", "hxxp", "spaced", "zero_width")
LEET_MAP = {
    "free": "fr33", "money": "m0ney", "paypal": "p4ypal", "click": "cl1ck",
    "prize": "pr1ze", "win": "w1n", "gift": "g1ft", "bank": "b4nk",
    "crypto": "crypt0", "verify": "ver1fy", "account": "acc0unt",
    "urgent": "ur9ent", "claim": "cl4im", "bonus": "b0nus",
}


@dataclass
class SynthRow:
    text: str
    label: int
    language: str
    category: str
    group: str
    source: str = "synthetic"


def _safe_format(template: str, slots: dict[str, str]) -> str:
    class _Safe(dict):
        def __missing__(self, key):
            return ""

    return template.format_map(_Safe(slots)).strip()


def _opener_prefix(opener: str, lang: str) -> str:
    if not opener:
        return ""
    if opener[-1] in ":,.!?;،।":
        return opener
    return opener + ("،" if lang == "Arabic" else ",")


def _base_slots(lang: str, rng: random.Random) -> dict[str, str]:
    pool = SLOTS[lang]
    slots = {key: rng.choice(values) for key, values in pool.items()}
    slots["amount_value"] = slots.get("amount", "")
    if rng.random() < 0.35:
        slots["amount"] = rng.choice(pool["amount"])
    return slots


def obfuscate_url(url: str, rng: random.Random) -> str:
    choice = rng.choice(SCAM_URL_OBFUSCATIONS)
    if choice == "bracket":
        return url.replace(".", "[.]", 1).replace("http://", "hxxp://")
    if choice == "paren":
        return url.replace(".", "(.)", 1)
    if choice == "dotword":
        return url.replace(".", " dot ", 1)
    if choice == "hxxp":
        return url.replace("https://", "hxxps://").replace("http://", "hxxp://")
    if choice == "spaced":
        parts = url.split("://", 1)
        body = parts[-1]
        if "." in body:
            domain, rest = body.split(".", 1)
            if len(domain) >= 4:
                spaced = ".".join(domain[:6])
                return (parts[0] + "://" if len(parts) == 2 else "") + spaced + "." + rest
        return url
    if choice == "zero_width":
        return ZERO_WIDTH.join(url)
    return url


def _leet_terms(text: str, rng: random.Random) -> str:
    words = text.split()
    for i, word in enumerate(words):
        key = re.sub(r"[^a-zA-Z]", "", word).lower()
        if key in LEET_MAP and rng.random() < 0.7:
            words[i] = word.replace(word.strip(".,:;!?"), LEET_MAP[key], 1)
    return " ".join(words)


def _insert_zero_width(text: str, rng: random.Random) -> str:
    if len(text) < 8:
        return text
    positions = sorted(rng.sample(range(1, len(text)), k=min(3, len(text) // 8)))
    out = []
    last = 0
    for pos in positions:
        out.append(text[last:pos])
        out.append(ZERO_WIDTH)
        last = pos
    out.append(text[last:])
    return "".join(out)


def _materialize(
    lang: str,
    label: int,
    category: str,
    body: str,
    body_index: int,
    rng: random.Random,
    obfuscation_rate: float,
    zero_width_rate: float,
    leet_rate: float,
) -> SynthRow:
    slots = _base_slots(lang, rng)
    if label == 1:
        opener = rng.choice(SCAM_OPENERS[lang])
        cta = rng.choice(SCAM_CTAS[lang])
        prefix = _opener_prefix(opener, lang) if rng.random() < 0.55 else ""
        suffix = _safe_format(cta, slots) if rng.random() < 0.7 else ""
        template = body
        if "{url}" in template and rng.random() < obfuscation_rate:
            slots["url"] = obfuscate_url(slots["url"], rng)
        if "{email}" in template and rng.random() < obfuscation_rate:
            slots["email"] = obfuscate_url(slots["email"], rng)
        text = " ".join(part for part in (prefix, _safe_format(template, slots), suffix) if part)
        if rng.random() < leet_rate:
            text = _leet_terms(text, rng)
        if rng.random() < zero_width_rate:
            text = _insert_zero_width(text, rng)
        if rng.random() < 0.18:
            text = text.upper()
        group = f"synth:{lang}:1:{category}:{body_index}"
        return SynthRow(text=text, label=1, language=lang, category=category, group=group)
    else:
        opener = rng.choice(HAM_OPENERS[lang])
        closer = rng.choice(HAM_CLOSERS[lang])
        prefix = _opener_prefix(opener, lang) if rng.random() < 0.35 else ""
        suffix = closer if rng.random() < 0.4 else ""
        text = " ".join(part for part in (prefix, _safe_format(body, slots), suffix) if part)
        group = f"synth:{lang}:0:{category}:{body_index}"
        return SynthRow(text=text, label=0, language=lang, category=category, group=group)


def _generate_for_language(
    lang: str,
    target_scam: int,
    target_ham: int,
    rng: random.Random,
    obfuscation_rate: float,
    zero_width_rate: float,
    leet_rate: float,
) -> list[SynthRow]:
    rows: list[SynthRow] = []
    scam_items = [(cat, i, body) for cat, bodies in SCAM_BODIES[lang].items() for i, body in enumerate(bodies)]
    ham_items = [(cat, i, body) for cat, bodies in HAM_BODIES[lang].items() for i, body in enumerate(bodies)]
    if target_scam > 0:
        indices = list(range(len(scam_items)))
        rng.shuffle(indices)
        n = 0
        rounds = 0
        while n < target_scam and rounds < max(4, target_scam // max(1, len(scam_items)) + 2):
            for idx in indices:
                if n >= target_scam:
                    break
                cat, i, body = scam_items[idx]
                rows.append(
                    _materialize(
                        lang, 1, cat, body, i, rng,
                        obfuscation_rate, zero_width_rate, leet_rate,
                    )
                )
                n += 1
            rounds += 1
    if target_ham > 0:
        indices = list(range(len(ham_items)))
        rng.shuffle(indices)
        n = 0
        rounds = 0
        while n < target_ham and rounds < max(4, target_ham // max(1, len(ham_items)) + 2):
            for idx in indices:
                if n >= target_ham:
                    break
                cat, i, body = ham_items[idx]
                rows.append(
                    _materialize(
                        lang, 0, cat, body, i, rng,
                        obfuscation_rate, zero_width_rate, leet_rate,
                    )
                )
                n += 1
            rounds += 1
    return rows


def _generate_mixed(
    n_scam: int,
    n_ham: int,
    rng: random.Random,
    obfuscation_rate: float,
    zero_width_rate: float,
    leet_rate: float,
) -> list[SynthRow]:
    rows: list[SynthRow] = []
    pairs = [
        ("English", "Bengali"), ("English", "Hindi"), ("English", "Arabic"),
        ("English", "Spanish"), ("English", "French"), ("Bengali", "Hindi"),
    ]
    for label, target, source_bodies in (
        (1, n_scam, SCAM_BODIES),
        (0, n_ham, HAM_BODIES),
    ):
        n = 0
        rounds = 0
        while n < target and rounds < target + 4:
            lang_a, lang_b = rng.choice(pairs)
            source_lang = lang_a if rng.random() < 0.5 else lang_b
            categories = list(source_bodies[source_lang].keys())
            category = rng.choice(categories)
            bodies = source_bodies[source_lang][category]
            body_index = rng.randrange(len(bodies))
            body = bodies[body_index]
            primary = source_lang
            other = lang_b if source_lang == lang_a else lang_a
            slots = _base_slots(primary, rng)
            if label == 1:
                opener = _opener_prefix(rng.choice(SCAM_OPENERS[primary]), primary)
                cta = rng.choice(SCAM_CTAS[other])
                if "{url}" in body and rng.random() < obfuscation_rate:
                    slots["url"] = obfuscate_url(slots["url"], rng)
                text = " ".join(
                    part for part in (
                        opener if rng.random() < 0.6 else "",
                        _safe_format(body, slots),
                        _safe_format(cta, slots) if rng.random() < 0.7 else "",
                    ) if part
                )
                if rng.random() < leet_rate:
                    text = _leet_terms(text, rng)
                if rng.random() < zero_width_rate:
                    text = _insert_zero_width(text, rng)
                rows.append(
                    SynthRow(
                        text=text, label=1, language="mixed",
                        category=f"mixed_{category}",
                        group=f"synth:mixed:1:{category}:{body_index}:{primary}",
                    )
                )
            else:
                opener = _opener_prefix(rng.choice(HAM_OPENERS[primary]), primary)
                closer = rng.choice(HAM_CLOSERS[other])
                text = " ".join(
                    part for part in (
                        _safe_format(body, slots),
                        opener if rng.random() < 0.3 else "",
                        closer if rng.random() < 0.35 else "",
                    ) if part
                )
                rows.append(
                    SynthRow(
                        text=text, label=0, language="mixed",
                        category=f"mixed_{category}",
                        group=f"synth:mixed:0:{category}:{body_index}:{primary}",
                    )
                )
            n += 1
            rounds += 1
    return rows


def generate_synthetic(config: dict, seed: int = 42) -> list[SynthRow]:
    cfg = config["data"]["synthetic"]
    rng = random.Random(seed)
    rows: list[SynthRow] = []
    for lang in cfg["languages"]:
        rows.extend(
            _generate_for_language(
                lang,
                int(cfg["per_language_scam"]),
                int(cfg["per_language_ham"]),
                rng,
                float(cfg.get("obfuscation_rate", 0.32)),
                float(cfg.get("zero_width_rate", 0.06)),
                float(cfg.get("leet_rate", 0.08)),
            )
        )
    rows.extend(
        _generate_mixed(
            int(cfg.get("mixed_scam", 500)),
            int(cfg.get("mixed_ham", 350)),
            rng,
            float(cfg.get("obfuscation_rate", 0.32)),
            float(cfg.get("zero_width_rate", 0.06)),
            float(cfg.get("leet_rate", 0.08)),
        )
    )
    return rows


def _fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    }


def validate_templates() -> list[str]:
    problems: list[str] = []
    for lang, categories in SCAM_BODIES.items():
        slot_keys = set(SLOTS[lang])
        for cta in SCAM_CTAS[lang]:
            missing = _fields(cta) - slot_keys
            if missing:
                problems.append(f"cta {lang}: missing slots {sorted(missing)}")
        for category, bodies in categories.items():
            for i, body in enumerate(bodies):
                missing = _fields(body) - slot_keys
                if missing:
                    problems.append(f"scam {lang}/{category}/{i}: missing slots {sorted(missing)}")
                    continue
                try:
                    rendered = _safe_format(body, _base_slots(lang, random.Random(1)))
                except Exception as exc:
                    problems.append(f"scam {lang}/{category}/{i}: {exc}")
                    continue
                if "{" in rendered or "}" in rendered:
                    problems.append(f"scam {lang}/{category}/{i}: unresolved placeholder {rendered[:120]}")
    for lang, categories in HAM_BODIES.items():
        slot_keys = set(SLOTS[lang])
        for category, bodies in categories.items():
            for i, body in enumerate(bodies):
                missing = _fields(body) - slot_keys
                if missing:
                    problems.append(f"ham {lang}/{category}/{i}: missing slots {sorted(missing)}")
                    continue
                try:
                    rendered = _safe_format(body, _base_slots(lang, random.Random(1)))
                except Exception as exc:
                    problems.append(f"ham {lang}/{category}/{i}: {exc}")
                    continue
                if "{" in rendered or "}" in rendered:
                    problems.append(f"ham {lang}/{category}/{i}: unresolved placeholder {rendered[:120]}")
    return problems
