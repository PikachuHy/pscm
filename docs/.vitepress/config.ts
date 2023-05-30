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
      {icon: 'github', link: 'https://github.com/PikachuHy/pscm'}
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
    {text: 'Guide', link: '/cn/intro', activeMatch: '/cn/'},
    {
      text: "Language",
      items: [
        {
          text: "简体中文", link: '/cn/intro'
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
      text: 'pscm',
      collapsible: true,
      items: [
        {
          text: 'pscm 简介',
          link: '/cn/intro'
        },
        {
          text: '如何构建 pscm',
          link: '/cn/how_to_build_pscm'
        },
        {
          text: 'pscm 数据类型',
          link: '/cn/cell'
        },
        {
          text: 'Equivalence predicates',
          link: '/cn/eq_eqv_equal'
        },
        {
          text: 'Register Machine',
          link: '/cn/register_machine'
        },
        {
          text: 'Continuation',
          link: '/cn/continuation'
        },
        {
          text: 'R4RS Support',
          link: '/cn/r4rs_support'
        },
      ]
    },
    {
      text: 'pscm-build',
      collapsible: true,
      items: [
        {
          text: 'pscm-build 简介',
          link: '/cn/pscm_build'
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
    }
  ]
}
