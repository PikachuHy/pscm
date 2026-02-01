export default {
  lang: 'en-US',
  title: "PikachuHy's Scheme",
  description: 'a scheme language implementation',
  base: '/',
  lastUpdated: true,
  ignoreDeadLinks: false,
  outDir: "public",
  locales: {
    "/cn/": {
      lang: 'zh-CN',
      title: 'pscm',
      description: '一个 scheme 语言实现',
    },
  },
  head: [],

  themeConfig: {
    nav: nav(),

    sidebar: {
      '/cn/': sidebarGuideZh(),
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/PikachuHy/pscm' }
    ],

    footer: {
      message: 'This website is released under the MIT License.',
      copyright: 'Copyright © 2022 PikachuHy'
    },

    editLink: {
      pattern: 'https://github.com/PikachuHy/pscm/edit/master/docs/:path'
    }
  }
}

function nav() {
  return [
    { text: 'Guide', link: '/cn/pscm_cc', activeMatch: '/cn/' },
    {
      text: "Language",
      items: [
        {
          text: "简体中文", link: '/cn/pscm_cc'
        }
      ]
    },
    {
      text: 'Github Issues',
      link: 'https://github.com/PikachuHy/pscm/issues'
    }
  ]
}

function sidebarGuideZh() {
  return [
    {
      text: 'pscm-cc (active development)',
      collapsible: true,
      items: [
        {
          text: 'pscm-cc 简介',
          link: '/cn/pscm_cc'
        },
        {
          text: 'pscm-cc 功能现状与开发规划',
          link: '/cn/pscm_cc_roadmap'
        },
        {
          text: 'pscm_cc 库 API 使用指南',
          link: '/cn/library_api'
        },
        {
          text: 'pscm-cc 字符支持实现说明',
          link: '/cn/char_support'
        },
        {
          text: 'pscm-cc 浮点数支持实现说明',
          link: '/cn/float_number_support'
        },
        {
          text: 'pscm-cc 点对支持实现说明',
          link: '/cn/dotted_pair_support'
        },
        {
          text: 'pscm-cc dynamic-wind 实现方案',
          link: '/cn/dynamic_wind_support'
        },
        {
          text: 'pscm-cc eval 实现方案',
          link: '/cn/eval_implementation'
        },
        {
          text: 'pscm-cc continuation 实现方案',
          link: '/cn/continuation_implementation'
        },
        {
          text: 'pscm-cc map 实现方案',
          link: '/cn/map_implementation'
        },
        {
          text: 'pscm-cc for-each 实现方案',
          link: '/cn/for_each_implementation'
        },
        {
          text: 'pscm-cc values 实现方案',
          link: '/cn/values_implementation'
        },
        {
          text: 'pscm-cc load 实现方案',
          link: '/cn/load_implementation'
        },
        {
          text: 'pscm-cc delay 实现方案',
          link: '/cn/delay_force_implementation'
        },
        {
          text: 'pscm-cc module 实现方案',
          link: '/cn/module_implementation'
        },
        {
          text: 'pscm-cc read options 实现方案',
          link: '/cn/read_options'
        },
        {
          text: 'pscm-cc debug options 实现方案',
          link: '/cn/debug_options'
        },
        {
          text: 'pscm-cc soft port 实现方案',
          link: '/cn/soft_port'
        },
        {
          text: 'pscm-cc 错误处理改进',
          link: '/cn/error_handling_improvements'
        },
        {
          text: 'pscm-cc wrong-number-of-args 错误处理',
          link: '/cn/wrong_number_of_args'
        }
      ]
    },
    {
      text: 'pscm v2 (archived)',
      collapsible: true,
      items: [
        {
          text: 'pscm v2 文档汇总（已归档）',
          link: '/cn/pscm_v2_archived'
        },
      ]
    },
    {
      text: 'misc',
      collapsible: true,
      items: [
        {
          text: '代码统计',
          link: '/cn/code_statistics'
        }
      ]
    },
  ]
}
