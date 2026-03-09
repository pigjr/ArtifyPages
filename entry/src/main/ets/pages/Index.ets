import { FileUtils, BookContent } from '../utils/FileUtils';

@Entry
@Component
struct Index {
  @State currentIndex: number = 0;
  @State selectedBook: string = '';
  @State bookContent: BookContent | null = null;
  @State isLoading: boolean = false;

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
    .borderBottom({ width: 1, color: '#E0E0E0' })
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
      
      if (this.isLoading) {
        Column() {
          Text('正在读取文件...')
            .fontSize(16)
            .fontColor('#666666')
          Progress({
            value: 0,
            total: 100,
            type: ProgressType.Linear
          })
          .width('80%')
          .margin({ top: 16 })
        }
        .width('100%')
        .padding(32)
        .backgroundColor('#FAFAFA')
        .borderRadius(12)
        .justifyContent(FlexAlign.Center)
        .alignItems(HorizontalAlign.Center)
      } else if (this.bookContent) {
        this.BookInfoDisplay()
      } else {
        this.FileImportInterface()
      }
    }
    .width('100%')
    .height('100%')
    .padding(24)
  }

  @Builder FileImportInterface() {
    Column({ space: 16 }) {
      Text('支持的格式: TXT')
        .fontSize(14)
        .fontColor('#666666')
      
      Text('文件大小建议: < 50MB（大文件会自动分块读取）')
        .fontSize(12)
        .fontColor('#999999')
        .margin({ top: 8 })
      
      Row({ space: 16 }) {
        Button('选择TXT文件')
          .backgroundColor('#2E7D32')
          .fontSize(16)
          .width(200)
          .height(48)
          .margin({ top: 16 })
          .onClick(() => {
            this.selectAndReadFile();
          })
        
        Button('示例文本')
          .backgroundColor('#FFFFFF')
          .fontColor('#2E7D32')
          .fontSize(16)
          .border({ width: 1, color: '#2E7D32' })
          .width(200)
          .height(48)
          .margin({ top: 16 })
          .onClick(() => {
            this.loadExampleText();
          })
      }
    }
    .width('100%')
    .padding(32)
    .backgroundColor('#FAFAFA')
    .borderRadius(12)
    .justifyContent(FlexAlign.Center)
    .alignItems(HorizontalAlign.Center)
  }

  @Builder BookInfoDisplay() {
    Column({ space: 16 }) {
      Text('书籍信息')
        .fontSize(18)
        .fontWeight(FontWeight.Medium)
        .fontColor('#333333')
      
      Row() {
        Text('书名:')
          .fontSize(14)
          .fontColor('#666666')
        Text(this.bookContent!.title)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#333333')
          .margin({ left: 8 })
      }
      
      Row() {
        Text('章节数:')
          .fontSize(14)
          .fontColor('#666666')
        Text(`${this.bookContent!.chapters.length} 章`)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#333333')
          .margin({ left: 8 })
      }
      
      Row() {
        Text('总字数:')
          .fontSize(14)
          .fontColor('#666666')
        Text(`${this.bookContent!.content.length.toLocaleString()} 字`)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#333333')
          .margin({ left: 8 })
      }
      
      Row({ space: 16 }) {
        Button('开始生成插图')
          .backgroundColor('#2E7D32')
          .fontSize(14)
          .padding({ left: 24, right: 24, top: 12, bottom: 12 })
          .onClick(() => {
            this.currentIndex = 2;
          })
        
        Button('重新选择')
          .backgroundColor('#FFFFFF')
          .fontColor('#666666')
          .fontSize(14)
          .border({ width: 1, color: '#E0E0E0' })
          .padding({ left: 24, right: 24, top: 12, bottom: 12 })
          .onClick(() => {
            this.bookContent = null;
            this.selectedBook = '';
          })
      }
      .margin({ top: 24 })
    }
    .width('100%')
    .padding(32)
    .backgroundColor('#FAFAFA')
    .borderRadius(12)
    .justifyContent(FlexAlign.Start)
    .alignItems(HorizontalAlign.Start)
  }

  private async selectAndReadFile() {
    try {
      this.isLoading = true;
      
      // TODO: 实际的文件选择器需要使用系统API
      // 这里先用模拟路径
      const mockFilePath = '/data/storage/el2/base/haps/entry/files/sample.txt';
      
      // 检查文件是否存在
      try {
        const fileInfo = await FileUtils.getFileInfo(mockFilePath);
        console.log('文件信息:', fileInfo);
        
        if (fileInfo.type.toLowerCase() !== 'txt') {
          throw new Error('仅支持TXT文件');
        }
        
        const content = await FileUtils.readTxtFile(mockFilePath);
        this.bookContent = content;
        this.selectedBook = content.title;
        console.log('文件读取成功:', content.title);
      } catch (fileError) {
        console.log('文件不存在，使用示例文本');
        this.loadExampleText();
      }
    } catch (error) {
      console.error('读取文件失败:', error);
      // TODO: 显示错误提示
    } finally {
      this.isLoading = false;
    }
  }

  private loadExampleText() {
    const exampleContent = `第一章 系统激活

林凡睁开眼睛，发现自己躺在陌生的房间里。周围是纯白色的墙壁，空气中弥漫着消毒水的味道。

"这是哪里？"他挣扎着坐起来，感觉头痛欲裂。

就在这时，一道机械的声音在脑海中响起：

【AI插图生成系统已激活】
【当前宿主：林凡】
【任务：为文学作品生成精美插图】

林凡愣住了，这是什么情况？

第二章 第一个任务

"欢迎来到艺绘书页系统，"那个声音继续说道，"我是你的助手小艺。"

"艺绘书页？这是什么？"林凡疑惑地问道。

"这是一个专为文学作品服务的AI插图生成系统。我可以分析文本内容，为每个场景生成对应的插图。"

林凡眼前一亮，这听起来很有趣。

"那么，我们开始吧！"他兴奋地说道。`;
    
    const mockBookContent: BookContent = {
      title: '艺绘书页系统',
      content: exampleContent,
      chapters: [
        {
          title: '第一章 系统激活',
          content: exampleContent.split('第二章')[0].trim(),
          startIndex: 0,
          endIndex: exampleContent.split('第二章')[0].length
        },
        {
          title: '第二章 第一个任务', 
          content: '第二章' + exampleContent.split('第二章')[1],
          startIndex: exampleContent.split('第二章')[0].length,
          endIndex: exampleContent.length
        }
      ]
    };
    
    this.bookContent = mockBookContent;
    this.selectedBook = mockBookContent.title;
    console.log('示例文本加载成功');
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