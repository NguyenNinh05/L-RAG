import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

const resources = {
  vi: {
    translation: {
      app: {
        title: 'LegalDiff',
        tagline: 'So sánh văn bản pháp luật với bằng chứng xác thực',
      },
      nav: {
        dashboard: 'Tổng quan',
        library: 'Thư viện',
        compare: 'So sánh mới',
        settings: 'Cài đặt',
        logout: 'Đăng xuất',
      },
      auth: {
        login: 'Đăng nhập',
        register: 'Đăng ký',
        email: 'Email',
        username: 'Tên đăng nhập',
        password: 'Mật khẩu',
      },
    },
  },
  en: {
    translation: {
      app: {
        title: 'LegalDiff',
        tagline: 'Legal document comparison with verified evidence',
      },
      nav: {
        dashboard: 'Dashboard',
        library: 'Library',
        compare: 'New Comparison',
        settings: 'Settings',
        logout: 'Log out',
      },
      auth: {
        login: 'Log in',
        register: 'Register',
        email: 'Email',
        username: 'Username',
        password: 'Password',
      },
    },
  },
}

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'vi',
    interpolation: {
      escapeValue: false,
    },
  })

export default i18n
