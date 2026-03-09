import { BusinessError } from '@kit.BasicServicesKit';
import { fileIo } from '@kit.CoreFileKit';
import { util } from '@kit.ArkTS';
export interface BookContent {
  title: string;
  content: string;
  chapters: Chapter[];
}

export interface Chapter {
  title: string;
  content: string;
  startIndex: number;
  endIndex: number;
}
interface FileInfo {
  name: string;
  size: number;
  type: string;
}

export class FileUtils {

  /**
   * 读取TXT文件内容
   * 对于大文件，建议分块读取以优化内存使用
   */
  static async readTxtFile(filePath: string): Promise<BookContent> {
    try {
      const file = fileIo.openSync(filePath, fileIo.OpenMode.READ_ONLY);
      const stat = fileIo.statSync(file.fd);
      const fileSize = stat.size;

      console.log(`文件大小: ${fileSize} bytes`);

      // 判断文件大小，决定读取策略
      if (fileSize > 10 * 1024 * 1024) { // 10MB
        console.warn('文件较大，建议分块读取');
        // return await FileUtils.readLargeTxtFile(file, fileSize);
        return await FileUtils.readSmallTxtFile(file, fileSize);
      } else {
        return await FileUtils.readSmallTxtFile(file, fileSize);
      }
    } catch (error) {
      const err = error as BusinessError;
      console.error('读取文件失败:', err.message);
      throw new Error(`读取文件失败: ${err.message}`);
    }
  }

  /**
   * 读取小文件（<10MB）- 一次性读入内存
   */

  private static async readSmallTxtFile(file: fileIo.File, fileSize: number): Promise<BookContent> {
    try {
      const buffer = new ArrayBuffer(fileSize);
      const readResult = await fileIo.read(file.fd, buffer); // 使用异步读取API
      if (readResult !== fileSize) {
        throw new Error(`读取不完整，预期 ${fileSize} 字节，实际读取 ${readResult} 字节`);
      }
      let uint8Data = new Uint8Array(buffer);
      //  使用decodeToString
      let textDecoderOptions: util.TextDecoderOptions = {
        fatal: false,
        ignoreBOM: true
      }
      let decodeToStringOptions: util.DecodeToStringOptions = {
        stream: false
      };
      let textDecoder = util.TextDecoder.create('utf-8', textDecoderOptions);
      let decodeToStr = textDecoder.decodeToString(uint8Data, decodeToStringOptions);
      console.info("tran decodeToString = " + decodeToStr); // tran decodeToString = 俩毛豆

      await fileIo.close(file); // 异步关闭文件描述符
      return FileUtils.parseBookContent(decodeToStr);
    } catch (err) {
      await fileIo.close(file).catch(() => {
      }); // 确保异常时仍关闭文件
      throw new Error(`读取文件失败: ${err.message}`);
    }
  }


  /**
   * 读取大文件（>10MB）- 分块读取
   */
  //   private static async readLargeTxtFile(file: fileIo.File, fileSize: number): Promise<BookContent> {
  //     const chunkSize = 1024 * 1024; // 1MB per chunk
  //     let fullContent = '';
  //     let offset = 0;
  //     let uint8Data = new Uint8Array( [228,191,169,230,175,155,232,177,134]);
  //
  //     while (offset < fileSize) {
  //       const remainingSize = fileSize - offset;
  //       const currentChunkSize = Math.min(chunkSize, remainingSize);
  //       const buffer = new ArrayBuffer(currentChunkSize);
  //       const readSize = fileIo.readSync(file.fd, buffer, { offset });
  //
  //       const decoder = new TextDecoder('utf-8'); // 明确指定编码格式
  //       const chunk = decoder.decode(new Uint8Array(buffer));
  //       fullContent += chunk;
  //
  //       offset += readSize;
  //
  //       console.log(`已读取: ${offset}/${fileSize} bytes (${Math.round(offset/fileSize*100)}%)`);
  //     }
  //
  //     fileIo.closeSync(file);
  //     return FileUtils.parseBookContent(fullContent);
  //   }

  /**
   * 解析书籍内容，识别章节
   */
  private static parseBookContent(content: string): BookContent {
    const chapters: Chapter[] = [];

    // 常见的章节标题正则表达式
    const chapterPatterns = [
      /第[一二三四五六七八九十百千万\d]+章[\s\S]*?\n/g,
      /Chapter\s*\d+[\s\S]*?\n/gi,
      /第\d+节[\s\S]*?\n/g,
      /\*\*\*第[一二三四五六七八九十百千万\d]+章\*\*\*[\s\S]*?\n/g
    ];

    let currentIndex = 0;
    let foundChapters = false;

    // 尝试匹配章节
    for (const pattern of chapterPatterns) {
      const matches = content.match(pattern);
      if (matches && matches.length > 1) {
        foundChapters = true;

        matches.forEach((match, index) => {
          const startPos = content.indexOf(match, currentIndex);
          const title = match.trim();
          const endPos = index < matches.length - 1 ?
            content.indexOf(matches[index + 1], startPos) :
            content.length;

          const chapterContent = content.substring(startPos, endPos).trim();

          chapters.push({
            title: title.replace(/\n/g, ''),
            content: chapterContent,
            startIndex: startPos,
            endIndex: endPos
          });

          currentIndex = endPos;
        });
        break;
      }
    }

    // 如果没有找到章节，将整个内容作为一章
    if (!foundChapters) {
      chapters.push({
        title: '全文',
        content: content,
        startIndex: 0,
        endIndex: content.length
      });
    }

    // 提取书名（从第一行或文件名）
    const firstLine = content.split('\n')[0].trim();
    const title = firstLine.length > 0 && firstLine.length < 50 ? firstLine : '未命名书籍';

    return {
      title,
      content,
      chapters
    };
  }

  /**
   * 获取文件信息
   */
  static async getFileInfo(filePath: string): Promise<FileInfo> {
    try {
      const file = fileIo.openSync(filePath, fileIo.OpenMode.READ_ONLY);
      const stat = fileIo.statSync(file.fd);
      const fileName = filePath.split('/').pop() || 'unknown';
      const fileType = fileName.split('.').pop() || '';

      fileIo.closeSync(file);

      return {
        name: fileName,
        size: stat.size,
        type: fileType
      };
    } catch (error) {
      console.error('获取文件信息失败:', error);
      let err = new Error("文件不存在"); // ✅ 使用Error对象
      throw err;
    }
  }

  /**
   * 格式化文件大小
   */
  static formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}
