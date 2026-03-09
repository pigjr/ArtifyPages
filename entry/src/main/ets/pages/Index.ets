@Entry
@Component
struct Index {
  @State currentIndex: number = 0;
  @State selectedBook: string = '';

  build() {
    Row() {
      // 侧边栏导航
      Column() {
        this.NavigationBar()
      }
      .width(240)
      .backgroundColor('#F5F5F5')
      .height('100%')

      // 主内容区域
      Column() {
        this.Header()
        
        if (this.currentIndex === 0) {
          this.HomePage()
        } else if (this.currentIndex === 1) {
          this.BookImportPage()
        } else if (this.currentIndex === 2) {
          this.AIGeneratePage()
        } else if (this.currentIndex === 3) {
          this.GalleryPage()
        } else if (this.currentIndex === 4) {
          this.SettingsPage()
        }
      }
      .layoutWeight(1)
      .height('100%')
    }
    .height('100%')
    .backgroundColor('#FFFFFF')
  }

  @Builder NavigationBar() {
    Column({ space: 8 }) {
      // Logo区域
      Row() {
        Text('艺绘书页')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor('#2E7D32')
      }
      .width('100%')
      .justifyContent(FlexAlign.Center)
      .padding(20)
      .backgroundColor('#FFFFFF')
      .borderRadius(8)
      .margin({ top: 16, bottom: 16 })

      // 导航菜单
      Column({ space: 4 }) {
        this.NavItem(0, '首页', '🏠')
        this.NavItem(1, '导入图书', '📚')
        this.NavItem(2, 'AI插图', '🎨')
        this.NavItem(3, '插图库', '🖼️')
        this.NavItem(4, '设置', '⚙️')
      }
      .width('100%')
      .padding({ left: 16, right: 16 })

      Blank()
    }
    .width('100%')
    .height('100%')
    .padding(16)
  }

  @Builder NavItem(index: number, title: string, icon: string) {
    Row() {
      Text(icon)
        .fontSize(18)
        .margin({ right: 12 })
      Text(title)
        .fontSize(14)
        .fontColor(this.currentIndex === index ? '#2E7D32' : '#666666')
    }
    .width('100%')
    .padding(12)
    .backgroundColor(this.currentIndex === index ? '#E8F5E8' : 'transparent')
    .borderRadius(6)
    .onClick(() => {
      this.currentIndex = index;
    })
  }

  @Builder Header() {
    Row() {
      Column() {
        Text('艺绘书页')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .fontColor('#333333')
        Text('AI驱动的电子书插图生成工具')
          .fontSize(14)
          .fontColor('#666666')
          .margin({ top: 4 })
      }
      .alignItems(HorizontalAlign.Start)

      Blank()

      if (this.selectedBook) {
        Text(`当前: ${this.selectedBook}`)
          .fontSize(12)
          .fontColor('#2E7D32')
          .padding({ left: 12, right: 12, top: 6, bottom: 6 })
          .backgroundColor('#E8F5E8')
          .borderRadius(16)
      }
    }
    .width('100%')
    .padding(24)
    .backgroundColor('#FFFFFF')
    .border({ bottom: { width: 1, color: '#E0E0E0' } })
  }

  @Builder HomePage() {
    Column({ space: 24 }) {
      // 欢迎卡片
      Column() {
        Text('欢迎使用艺绘书页')
          .fontSize(28)
          .fontWeight(FontWeight.Bold)
          .fontColor('#333333')
        Text('为您的电子书生成精美的AI插图')
          .fontSize(16)
          .fontColor('#666666')
          .margin({ top: 8 })
        
        Row({ space: 16 }) {
          Button('导入图书')
            .backgroundColor('#2E7D32')
            .fontSize(14)
            .padding({ left: 24, right: 24, top: 12, bottom: 12 })
            .onClick(() => {
              this.currentIndex = 1;
            })
          
          Button('开始创作')
            .backgroundColor('#FFFFFF')
            .fontColor('#2E7D32')
            .fontSize(14)
            .border({ width: 1, color: '#2E7D32' })
            .padding({ left: 24, right: 24, top: 12, bottom: 12 })
            .onClick(() => {
              this.currentIndex = 2;
            })
        }
        .margin({ top: 24 })
      }
      .width('100%')
      .padding(32)
      .backgroundColor('#FAFAFA')
      .borderRadius(12)

      // 快速操作
      Column({ space: 16 }) {
        Text('快速操作')
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .fontColor('#333333')
        
        Row({ space: 16 }) {
          this.QuickActionCard('📖', '打开最近图书', '继续上次的项目')
          this.QuickActionCard('🎨', '快速生成', '一键生成插图')
          this.QuickActionCard('📁', '插图库', '管理所有插图')
        }
        .width('100%')
      }
      .width('100%')
      .alignItems(HorizontalAlign.Start)

      Blank()
    }
    .width('100%')
    .height('100%')
    .padding(24)
  }

  @Builder QuickActionCard(icon: string, title: string, desc: string) {
    Column() {
      Text(icon)
        .fontSize(32)
        .margin({ bottom: 12 })
      Text(title)
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor('#333333')
      Text(desc)
        .fontSize(12)
        .fontColor('#666666')
        .margin({ top: 4 })
    }
    .width(160)
    .height(120)
    .padding(16)
    .backgroundColor('#FFFFFF')
    .borderRadius(8)
    .border({ width: 1, color: '#E0E0E0' })
    .justifyContent(FlexAlign.Center)
    .alignItems(HorizontalAlign.Center)
  }

  @Builder BookImportPage() {
    Column({ space: 24 }) {
      Text('导入电子书')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333333')
      
      Column({ space: 16 }) {
        Text('支持的格式: EPUB, PDF, TXT')
          .fontSize(14)
          .fontColor('#666666')
        
        Button('选择文件')
          .backgroundColor('#2E7D32')
          .fontSize(16)
          .width(200)
          .height(48)
          .margin({ top: 16 })
      }
      .width('100%')
      .padding(32)
      .backgroundColor('#FAFAFA')
      .borderRadius(12)
      .justifyContent(FlexAlign.Center)
      .alignItems(HorizontalAlign.Center)
    }
    .width('100%')
    .height('100%')
    .padding(24)
  }

  @Builder AIGeneratePage() {
    Column({ space: 24 }) {
      Text('AI插图生成')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333333')
      
      Column({ space: 16 }) {
        Text('选择文本段落并生成插图')
          .fontSize(14)
          .fontColor('#666666')
        
        Button('开始生成')
          .backgroundColor('#2E7D32')
          .fontSize(16)
          .width(200)
          .height(48)
          .margin({ top: 16 })
      }
      .width('100%')
      .padding(32)
      .backgroundColor('#FAFAFA')
      .borderRadius(12)
      .justifyContent(FlexAlign.Center)
      .alignItems(HorizontalAlign.Center)
    }
    .width('100%')
    .height('100%')
    .padding(24)
  }

  @Builder GalleryPage() {
    Column({ space: 24 }) {
      Text('插图库')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333333')
      
      Column({ space: 16 }) {
        Text('管理所有生成的插图')
          .fontSize(14)
          .fontColor('#666666')
      }
      .width('100%')
      .padding(32)
      .backgroundColor('#FAFAFA')
      .borderRadius(12)
      .justifyContent(FlexAlign.Center)
      .alignItems(HorizontalAlign.Center)
    }
    .width('100%')
    .height('100%')
    .padding(24)
  }

  @Builder SettingsPage() {
    Column({ space: 24 }) {
      Text('设置')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333333')
      
      Column({ space: 16 }) {
        Text('应用配置和偏好设置')
          .fontSize(14)
          .fontColor('#666666')
      }
      .width('100%')
      .padding(32)
      .backgroundColor('#FAFAFA')
      .borderRadius(12)
      .justifyContent(FlexAlign.Center)
      .alignItems(HorizontalAlign.Center)
    }
    .width('100%')
    .height('100%')
    .padding(24)
  }
}