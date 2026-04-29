/**
 * MumzSense Internationalisation
 * EN + AR translations for all UI text
 */

export const resources = {
  en: {
    translation: {
      // Landing
      tagline: "Ask anything about your baby's journey",
      subheadline: "Answers from mothers who've been exactly where you are",
      trustSignal: "Powered by 560+ real mother experiences",
      placeholder: "Ask about feeding, sleep, health, development...",
      askButton: "Ask",
      languageToggle: "عربي",
      
      // Stages
      stages: {
        label: "Baby's stage",
        trimester: "Trimester",
        newborn: "Newborn",
        "0-3m": "0–3 months",
        "3-6m": "3–6 months",
        "6-12m": "6–12 months",
        toddler: "Toddler",
      },
      
      // Topics
      topics: {
        feeding: "Feeding",
        sleep: "Sleep",
        health: "Health",
        development: "Development",
        gear: "Gear",
        postpartum: "Postpartum",
        mental_health: "Mental Health",
      },
      
      // Urgency badges
      urgency: {
        routine: "Routine",
        monitor: "Worth Monitoring",
        "seek-help": "Seek Medical Help",
      },
      
      // Chat
      answerBy: "MumzMind",
      cached: "Cached response",
      latency: "{{ms}}ms",
      sources: "{{count}} source",
      sources_plural: "{{count}} sources",
      showSources: "Show sources",
      hideSources: "Hide sources",
      trustScore: "Trust",
      similarity: "Match",
      
      // Secondary language toggle
      viewInArabic: "عرض بالعربية",
      viewInEnglish: "View in English",
      
      // Urgency CTA
      seekHelpTitle: "This sounds like something a healthcare professional should weigh in on.",
      seekHelpCTA: "Find a paediatrician near you",
      
      // Uncertainty
      uncertaintyTitle: "I don't have enough similar experiences to answer this confidently.",
      uncertaintyBody: "You might find better answers by speaking with your healthcare provider or visiting our community forum.",
      rephrase: "Try rephrasing",
      tryDifferentStage: "Try a different stage",
      
      // Error
      errorTitle: "Something went wrong",
      errorBody: "We couldn't get a response. Please try again in a moment.",
      retry: "Try again",
      
      // Feedback
      feedbackPrompt: "Was this helpful?",
      feedbackYes: "Yes, helpful",
      feedbackNo: "Not quite",
      feedbackThanks: "Thanks for your feedback!",
      
      // Health/Status
      systemOnline: "System online",
      systemDegraded: "Limited service",
      cacheHit: "⚡ Instant from cache",
      
      // About
      aboutTitle: "About MumzSense",
      aboutDesc: "A bilingual maternal Q&A assistant trained on curated community experiences.",
      corpusStats: "Knowledge Base",
      totalPosts: "mother experiences",
      
      // Input validation
      queryTooShort: "Please type at least 3 characters",
      queryTooLong: "Query too long (max 1000 characters)",
      
      // Confidence
      confidence: {
        high: "High confidence",
        medium: "Medium confidence",
        low: "Low confidence",
        none: "Uncertain",
        deferred: "Referred to professional",
      },
    },
  },
  
  ar: {
    translation: {
      // Landing
      tagline: "اسألي أي شيء عن رحلة طفلك",
      subheadline: "إجابات من أمهات مررن بنفس تجربتك تماماً",
      trustSignal: "مدعوم بأكثر من 560 تجربة حقيقية للأمهات",
      placeholder: "اسألي عن الرضاعة، النوم، الصحة، التطور...",
      askButton: "اسألي",
      languageToggle: "English",
      
      // Stages
      stages: {
        label: "مرحلة طفلك",
        trimester: "الحمل",
        newborn: "حديث الولادة",
        "0-3m": "0–3 أشهر",
        "3-6m": "3–6 أشهر",
        "6-12m": "6–12 شهر",
        toddler: "الطفل الصغير",
      },
      
      // Topics
      topics: {
        feeding: "الرضاعة",
        sleep: "النوم",
        health: "الصحة",
        development: "التطور",
        gear: "المستلزمات",
        postpartum: "ما بعد الولادة",
        mental_health: "الصحة النفسية",
      },
      
      // Urgency badges
      urgency: {
        routine: "روتيني",
        monitor: "يستحق المتابعة",
        "seek-help": "راجعي طبيباً",
      },
      
      // Chat
      answerBy: "MumzMind",
      cached: "استجابة محفوظة",
      latency: "{{ms}} مللي ثانية",
      sources: "{{count}} مصدر",
      sources_plural: "{{count}} مصادر",
      showSources: "إظهار المصادر",
      hideSources: "إخفاء المصادر",
      trustScore: "الثقة",
      similarity: "التطابق",
      
      viewInArabic: "عرض بالعربية",
      viewInEnglish: "View in English",
      
      seekHelpTitle: "يبدو أن هذا يستدعي رأي متخصص في الرعاية الصحية.",
      seekHelpCTA: "ابحثي عن طبيب أطفال قريب منكِ",
      
      uncertaintyTitle: "لا أملك تجارب مشابهة كافية للإجابة بثقة على هذا السؤال.",
      uncertaintyBody: "قد تجدين إجابات أفضل بالتحدث مع مقدم الرعاية الصحية أو زيارة منتدانا.",
      rephrase: "أعيدي الصياغة",
      tryDifferentStage: "جربي مرحلة مختلفة",
      
      errorTitle: "حدث خطأ ما",
      errorBody: "لم نتمكن من الحصول على رد. يرجى المحاولة مرة أخرى.",
      retry: "حاولي مجدداً",
      
      feedbackPrompt: "هل كان هذا مفيداً؟",
      feedbackYes: "نعم، مفيد",
      feedbackNo: "ليس تماماً",
      feedbackThanks: "شكراً على ملاحظاتك!",
      
      systemOnline: "النظام متاح",
      systemDegraded: "خدمة محدودة",
      cacheHit: "⚡ فوري من الذاكرة",
      
      aboutTitle: "عن MumzSense",
      aboutDesc: "مساعد Q&A للأمومة ثنائي اللغة مدرّب على تجارب مجتمعية منتقاة.",
      corpusStats: "قاعدة المعرفة",
      totalPosts: "تجربة أمومة",
      
      queryTooShort: "يرجى كتابة 3 أحرف على الأقل",
      queryTooLong: "الاستفسار طويل جداً (الحد الأقصى 1000 حرف)",
      
      confidence: {
        high: "ثقة عالية",
        medium: "ثقة متوسطة",
        low: "ثقة منخفضة",
        none: "غير متأكد",
        deferred: "تحويل للمتخصص",
      },
    },
  },
}

export const defaultNS = 'translation'
export const fallbackLng = 'en'
