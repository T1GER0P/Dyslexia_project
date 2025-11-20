import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Volume2, VolumeX, Type, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const sampleText = `Once upon a time, in a forest filled with tall trees and colorful flowers, there lived a curious little fox named Ruby. Ruby loved to explore and discover new things every day. One sunny morning, she found a mysterious path she had never seen before. With excitement in her heart, Ruby decided to follow the winding trail to see where it would lead.`;

const ReadingDemo = () => {
  const [fontSize, setFontSize] = useState([18]);
  const [lineSpacing, setLineSpacing] = useState([1.8]);
  const [isDyslexicFont, setIsDyslexicFont] = useState(false);
  const [isReading, setIsReading] = useState(false);
  const { toast } = useToast();

  const toggleReading = () => {
    if (!isReading) {
      toast({
        title: "Text-to-Speech Started",
        description: "Reading text aloud...",
      });
      // In a real app, this would trigger text-to-speech
    } else {
      toast({
        title: "Text-to-Speech Stopped",
        description: "Reading paused.",
      });
    }
    setIsReading(!isReading);
  };

  return (
    <section id="demo" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="mx-auto max-w-6xl">
          <div className="mb-12 text-center">
            <h2 className="mb-4 text-3xl font-bold md:text-4xl">
              Try the Reading Experience
            </h2>
            <p className="text-lg text-muted-foreground">
              Customize the text to find what works best for you
            </p>
          </div>

          <div className="grid gap-8 lg:grid-cols-[300px_1fr]">
            {/* Controls */}
            <Card className="p-6 shadow-medium h-fit">
              <div className="space-y-6">
                <div>
                  <div className="mb-4 flex items-center gap-2">
                    <Type className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold">Customize Reading</h3>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <Label className="mb-2 block">Font Size: {fontSize}px</Label>
                      <Slider
                        value={fontSize}
                        onValueChange={setFontSize}
                        min={14}
                        max={32}
                        step={2}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <Label className="mb-2 block">Line Spacing: {lineSpacing}x</Label>
                      <Slider
                        value={lineSpacing}
                        onValueChange={setLineSpacing}
                        min={1.2}
                        max={3}
                        step={0.2}
                        className="w-full"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label htmlFor="dyslexic-font">Dyslexic-Friendly Font</Label>
                      <Switch
                        id="dyslexic-font"
                        checked={isDyslexicFont}
                        onCheckedChange={setIsDyslexicFont}
                      />
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <Button
                    onClick={toggleReading}
                    className="w-full"
                    variant={isReading ? "secondary" : "default"}
                  >
                    {isReading ? (
                      <>
                        <VolumeX className="mr-2 h-4 w-4" />
                        Stop Reading
                      </>
                    ) : (
                      <>
                        <Volume2 className="mr-2 h-4 w-4" />
                        Read Aloud
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </Card>

            {/* Reading Area */}
            <Card className="p-8 shadow-medium">
              <div className="mb-4 flex items-center gap-2 text-accent">
                <Sparkles className="h-5 w-5" />
                <span className="text-sm font-medium">AI-Enhanced Reading</span>
              </div>
              
              <div
                className="prose prose-lg max-w-none"
                style={{
                  fontSize: `${fontSize[0]}px`,
                  lineHeight: lineSpacing[0],
                  fontFamily: isDyslexicFont ? "'Comic Sans MS', 'Arial', sans-serif" : "inherit",
                  letterSpacing: isDyslexicFont ? "0.05em" : "normal",
                }}
              >
                <p className="text-foreground transition-all duration-300">
                  {sampleText}
                </p>
              </div>

              {isReading && (
                <div className="mt-6 flex items-center gap-3 rounded-lg bg-primary/10 p-4">
                  <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
                  <span className="text-sm font-medium text-primary">Currently reading...</span>
                </div>
              )}
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ReadingDemo;
